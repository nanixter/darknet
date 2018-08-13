#ifndef DARKNET_WRAPPER_HPP
#define DARKNET_WRAPPER_HPP

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstring>

#include "darknetserver.pb.h"

// typedef enum {
//     CONVOLUTIONAL,
//     DECONVOLUTIONAL,
//     CONNECTED,
//     MAXPOOL,
//     SOFTMAX,
//     DETECTION,
//     DROPOUT,
//     CROP,
//     ROUTE,
//     COST,
//     NORMALIZATION,
//     AVGPOOL,
//     LOCAL,
//     SHORTCUT,
//     ACTIVE,
//     RNN,
//     GRU,
//     LSTM,
//     CRNN,
//     BATCHNORM,
//     NETWORK,
//     XNOR,
//     REGION,
//     YOLO,
//     REORG,
//     UPSAMPLE,
//     LOGXENT,
//     L2NORM,
//     BLANK
// } LAYER_TYPE;

extern "C" {
	// struct layer;
	// struct image;
	// struct detection;
	// struct network;
	// void cuda_set_device(int);
	// network *load_network(char *cfg, char *weights, int clear);
	// void set_batch_network(network *net, int b);
	// void rgbgr_image(image im);
	// image letterbox_image(image im, int w, int h);
	// float *network_predict(network *net, float *input);
	// void do_nms_obj(detection *dets, int total, int classes, float thresh);
	// void fill_cpu(int N, float ALPHA, float * X, int INCX);
	// void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
	// detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
	#undef __cplusplus
	#include "darknet.h"
	#define __cplusplus 1
}

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#endif

namespace DarknetWrapper {

	typedef struct
	{
		bool done;
		darknetServer::KeyFrame frame;
		darknetServer::DetectedObjects detectedObjects;
		void *tag;
	} WorkRequest;

	class DetectionQueue
	{
	public:

		void push_back(WorkRequest elem) {
			std::lock_guard<std::mutex> lock(this->mutex);
			this->queue.push(elem);
			this->cv.notify_one();
		}

		// TODO: Make this a counting cv instead so that we can batch images!
		void pop_front(WorkRequest &elem) {
			std::unique_lock<std::mutex> lock(this->mutex);
			cv.wait(lock, [this](){ return !this->queue.empty(); });

			// Once the cv wakes us up....
			if(!this->queue.empty()) {
				elem = this->queue.front();
				this->queue.pop();
			}
		}

	private:
		std::queue<WorkRequest> queue;
		std::mutex mutex;
		std::condition_variable cv;

	}; // class DetectionQueue


	class Detector {
	public:

		void Init(int argc, char** argv, DetectionQueue *requestQueue, DetectionQueue *completionQueue) {
			// Store pointers to the workQueues
			this->requestQueue = requestQueue;
			this->completionQueue = completionQueue;

			// Initialization: Load config files, labels, graph, etc.,
			// Config the GPU and get into a thread that is ready to accept
			// images for detection.
			cuda_set_device(0);
			char *datacfg = argv[1];
			char *cfgfile = argv[2];
			char *weightfile = argv[3];

			net = load_network(cfgfile, weightfile, 0);
			set_batch_network(net, 1);

			this->numNetworkOutputs = this.size_network();
    		this->predictions = new float[numNetworkOutputs];
    		this->average = new float[numNetworkOutputs];
		}

		void doDetection() {
			WorkRequest elem;
			float nms = .4;
			layer l = net->layers[net->n-1];
			image newImage;
			image newImage_letterboxed;
			detection *dets = nullptr;
			int nboxes = 0;

			while(true) {
				// Wait on the requestQueue
				requestQueue->pop_front(elem);

				// Convert to the right format
				// Allocate memory for data in 'image', based on the size of 'data' in frame
				newImage.data = new float[elem.frame.data_size()];

				// Copy from the frame in elem to the 'image' format that darknet uses internally...
				this->convertFrameToImage(&(elem.frame), &newImage);

				// Convert to the RGBGR format that YOLO operates on..
				rgbgr_image(newImage);

				// Add black borders (letter-boxing) around the image to ensure that the image
				// is of the correct width and height that YOLO expects.
				newImage_letterboxed = letterbox_image(newImage, net->w, net->h);

				/* Now we finally run the actual network	*/
				network_predict(net, newImage_letterboxed.data);
				this->remember_network();
				dets = this->average_predictions(&nboxes, newImage.h, newImage.w);

				// What the hell does this do?
				if (nms > 0) {
					do_nms_obj(dets, nboxes, l.classes, nms);
				}

				/* Copy detected objects to the WorkRequest */
				for (int i = 0; i < nboxes; i++) {
					if(dets[i].objectness == 0) continue;
					darknetServer::DetectedObjects_DetectedObject_box bbox;
					bbox.set_x(dets[i].bbox.x);
					bbox.set_y(dets[i].bbox.y);
					bbox.set_w(dets[i].bbox.w);
					bbox.set_h(dets[i].bbox.h);
					darknetServer::DetectedObjects_DetectedObject *object = elem.detectedObjects.add_objects();
					object->set_allocated_bbox(&bbox);
					object->set_objectness(dets[i].objectness);
					object->set_classes(dets[i].classes);
					object->set_sort_class(dets[i].sort_class);
					for (int j = 0; j < l.classes; j++)
						object->add_prob(dets[i].prob[j]);
				}

				// Put the result back on the completionQueue.
				completionQueue->push_back(elem);

				// Clean up
				free_detections(dets, nboxes);
				delete [] newImage.data;
			}
		}

		void Shutdown() {
			// Set locally owned pointers to NULL;
			this->requestQueue = nullptr;
			this->completionQueue = nullptr;

			// Free any darknet resources held. Close the GPU connection, etc...
			delete this->predictions;
			delete this->average;
		}

	private:
		void convertFrameToImage(darknetServer::KeyFrame *frame, image *newImage) {
			newImage->w = frame->width();
			newImage->h = frame->height();
			newImage->c = frame->numchannels();
			int dataSize = frame->data_size();
			for (int i = 0; i < dataSize; i++)
				newImage->data[i] = frame->data(i);
		}

		// Helper functions stolen from demo.c
		int size_network()
		{
			int count = 0;
			for(int i = 0; i < this->net->n; ++i){
				layer l = this->net->layers[i];
				if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
					count += l.outputs;
				}
			}
			return count;
		}
		void remember_network()
		{
			int count = 0;
			for(int i = 0; i < this->net->n; ++i){
				layer l = this->net->layers[i];
				if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
					std::memcpy(predictions + count, this->net->layers[i].output, sizeof(float) * l.outputs);
					count += l.outputs;
				}
			}
		}

		detection *average_predictions(int *nboxes, int height, int width)
		{
			int i, j;
			int count = 0;
			fill_cpu(this->numNetworkOutputs, 0, average, 1);
			axpy_cpu(this->numNetworkOutputs, 1./3, predictions, 1, average, 1);

			for(i = 0; i < this->net->n; ++i){
				layer l = this->net->layers[i];
				if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
					std::memcpy(l.output, average + count, sizeof(float) * l.outputs);
					count += l.outputs;
				}
			}
			return get_network_boxes(this->net, width, height, 0.5, 0.5, 0, 1, nboxes);
		}

		// All the darknet globals.
		DetectionQueue *requestQueue;
		DetectionQueue *completionQueue;
		float *predictions;
		float *average;
		char *datacfg;
		char *cfgfile;
		char *weightfile;

		network *net;
		int numNetworkOutputs;

	}; // class Detector

} // namespace DarknetWrapper

#endif // DARKNET_WRAPPER_CPP
