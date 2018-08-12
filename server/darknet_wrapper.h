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

extern "C" {
#include "darknet.h"
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
			std::lock_guard<std::mutex> lock(this.mutex);
			this.queue.push(elem);
			this.cv.notify_one();
		}

		// TODO: Make this a counting cv instead so that we can batch images!
		void pop_front(WorkRequest &elem) {
			std::unique_lock<std::mutex> lock(this.mutex);
			cv.wait(lock, !this.queue.empty());

			// Once the cv wakes us up....
			if(!this.queue.empty()) {
				elem = this.queue.front();
				this.queue.pop();
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
			this.requestQueue = requestQueue;
			this.completionQueue = completionQueue;

			// Initialization: Load config files, labels, graph, etc.,
			// Config the GPU and get into a thread that is ready to accept
			// images for detection.
			cuda_set_device(0);
			char *datacfg = argv[1];
			char *cfg = argv[2];
			char *weights = argv[3];
			list *options = read_data_cfg(datacfg);
			int classes = option_find_int(options, "classes", 20);
			char *name_list = option_find_str(options, "names", "data/names.list");
			char **names = get_labels(name_list);

			net = load_network(cfgfile, weightfile, 0);
			set_batch_network(net, 1);

			int numNetworkOutputs = size_network(net);
    		this.predictions = new float[numNetworkOutputs];

			// Incomplete
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
				/*------------------------------------------*/
				/* Wait on the requestQueue					*/
				/*------------------------------------------*/
				requestQueue.pop_front(elem);

				/*------------------------------------------*/
				/* Convert to the right format				*/
				/*------------------------------------------*/

				// Allocate memory for data in 'image', based on the size of 'data' in frame
				newImage.data = new float[elem.frame.data_size()];
				// Copy from the frame in elem to the 'image' format that darknet uses internally...
				this.convertFrameToImage(&(elem.frame), &newImage);
				// Convert to the RGBGR format that YOLO operates on..
				rgbgr_image(newImage);
				// Add black borders (letter-boxing) around the image to ensure that the image
				// is of the correct width and height that YOLO expects.
				newImage_letterboxed = letterbox_image(newImage, net->w, net->h);

				/*------------------------------------------*/
				/* Now we finally run the actual network	*/
				/*------------------------------------------*/
				network_predict(net, newImage_letterboxed.data);
				this.remember_network(net);
				dets = this.average_predictions(net, &nboxes, newImage.h, newImage.w);

				if (nms > 0) {
					do_nms_obj(dets, nboxes, l.classes, nms);
				}
// for(i = 0; i < nboxes; ++i){
//         if(dets[i].objectness == 0) continue;
//         box a = dets[i].bbox;
//         for(j = i+1; j < nboxes; ++j){
//             if(dets[j].objectness == 0) continue;
//             box b = dets[j].bbox;
//             if (box_iou(a, b) > thresh){
//                 dets[j].objectness = 0;
//                 for(k = 0; k < classes; ++k){
//                     dets[j].prob[k] = 0;
//                 }
//             }
//         }
//     }
				/*------------------------------------------*/
				/* Copy detected objects to the WorkRequest */
				/*------------------------------------------*/
				for (int i = 0; i < nboxes; i++) {
					if(dets[i].objectness == 0) continue;
					::darknetServer::DetectedObject *object = elem.detecteObjects.add_objects();

				}


				/*------------------------------------------*/
				// Put the result back on the completionQueue.
				/*------------------------------------------*/
				completionQueue.push_back(elem);

				/*------------------------------------------*/
				// Clean up
				/*------------------------------------------*/
				free_detections(dets, nboxes);
				delete [] newImage.data;
			}

		}

		void ShutDown() {
			// Set locally owned pointers to NULL;
			requestQueue = nullptr;
			completionQueue = nullptr;

			// Free any darknet resources held. Close the GPU connection, etc...
		}

	private:
		void convertFrameToImage(darknetServer::KeyFrame *frame, image *newImage) {
			newImage->w = frame->width();
			newImage->h = frame->height();
			newImage->c = frame->numchannels();
			int dataSize = frame.data_size();
			for (int i = 0; i < dataSize; i++)
				newImage->data[i] = frame->data(i);
		}

		// Helper functions stolen from demo.c
		int size_network(network *net)
		{
			int count = 0;
			for(int i = 0; i < net->n; ++i){
				layer l = net->layers[i];
				if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
					count += l.outputs;
				}
			}
			return count;
		}
		void remember_network(network *net)
		{
			int count = 0;
			for(int i = 0; i < net->n; ++i){
				layer l = net->layers[i];
				if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
					std::memcpy(predictions + count, net->layers[i].output, sizeof(float) * l.outputs);
					count += l.outputs;
				}
			}
		}

		detection *average_predictions(network *net, int *nboxes, int height, int width)
		{
			int i, j;
			int count = 0;
			fill_cpu(numNetworkOutputs, 0, average, 1);
			axpy_cpu(numNetworkOutputs, 1./3, predictions, 1, average, 1);

			for(i = 0; i < net->n; ++i){
				layer l = net->layers[i];
				if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
					std::memcpy(l.output, average + count, sizeof(float) * l.outputs);
					count += l.outputs;
				}
			}
			return get_network_boxes(net, width, height, 0.5, 0.5, 0, 1, nboxes);
		}

		// All the darknet globals.
		DetectionQueue *requestQueue;
		DetectionQueue *completionQueue;
		float *predictions;
		float *average;

		network *net;

	}; // class Detector

} // namespace DarknetWrapper

#endif // DARKNET_WRAPPER_CPP