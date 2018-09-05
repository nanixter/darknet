#ifndef DARKNET_WRAPPER_FB_HPP
#define DARKNET_WRAPPER_FB_HPP

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <sys/time.h>

extern "C" {
	#undef __cplusplus
	#include "darknet.h"
	#define __cplusplus 1
}

struct timestamp {
    struct timeval start;
    struct timeval end;
};

static inline void tvsub(struct timeval *x,
						 struct timeval *y,
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

void probe_time_start2(struct timestamp *ts)
{
    gettimeofday(&ts->start, NULL);
}

float probe_time_end2(struct timestamp *ts)
{
    struct timeval tv;
    gettimeofday(&ts->end, NULL);
	tvsub(&ts->end, &ts->start, &tv);
	return (tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0);
}

namespace DarknetWrapper {

	typedef struct
	{
		bool done;
		const darknetServer::KeyFrame *frame;
		flatbuffers::grpc::MessageBuilder *messageBuilder;
		flatbuffers::grpc::Message<DetectedObjects>* responseMessage;
		void *tag;
	} WorkRequest;

	/*
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
	*/

	class Detector {
	public:

		void Init(int argc, char** argv) {
			// Initialization: Load config files, labels, graph, etc.,
			// Config the GPU and get into a thread that is ready to accept
			// images for detection.
			#ifdef GPU
			cuda_set_device(0);
			#endif
			char *datacfg = argv[1];
			char *cfgfile = argv[2];
			char *weightfile = argv[3];

			this->net = load_network(cfgfile, weightfile, 0);
			set_batch_network(net, 1);

			this->numNetworkOutputs = this->size_network();
			this->predictions = new float[numNetworkOutputs];
			this->average = new float[numNetworkOutputs];
		}

		void Shutdown() {
			// Free any darknet resources held. Close the GPU connection, etc...
			delete this->predictions;
			delete this->average;
		}

		void doDetection(WorkRequest &elem) {
			float nms = .4;
			layer l = net->layers[net->n-1];
			detection *dets = nullptr;
			int nboxes = 0;

			image newImage;
			image newImage_letterboxed;

			std::cout << "doDetection: new requeust " << elem.tag << std::endl;
			probe_time_start2(&ts_detect);

			// Convert to the right format
			// Allocate memory for data in 'image', based on the size of 'data' in frame
			newImage.data = new float[elem.frame->data()->size()];

			// Copy from the frame in elem to the 'image' format that darknet uses internally...
			this->convertFrameToImage(elem.frame, &newImage);

			//save_image(newImage, "recieved");

			// Convert to the RGBGR format that YOLO operates on..
			rgbgr_image(newImage);

			// Add black borders (letter-boxing) around the image to ensure that the image
			// is of the correct width and height that YOLO expects.
			newImage_letterboxed = letterbox_image(newImage, net->w, net->h);

			//save_image(newImage_letterboxed, "letterboxed");

			/* Now we finally run the actual network	*/
			probe_time_start2(&ts_gpu);
			network_predict(net, newImage_letterboxed.data);
			this->remember_network();
			dets = this->average_predictions(&nboxes, newImage.h, newImage.w);

			// What the hell does this do?
			if (nms > 0) {
				do_nms_obj(dets, nboxes, l.classes, nms);
			}

			std::cout << elem.tag << " GPU processing took " << probe_time_end2(&ts_gpu) << " milliseconds"<< std::endl;

			//draw_detections(newImage_letterboxed, dets, nboxes, 0.5, NULL, NULL, l.classes);
			//save_image(newImage_letterboxed, "detected");

			/* Copy detected objects to the WorkRequest */
			std::vector<flatbuffers::Offset<DetectedObject>> objects;
			int numObjects = 0;
			for (int i = 0; i < nboxes; i++) {
				if(dets[i].objectness == 0) continue;
				bbox box(dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
				std::vector<float> prob;
				for (int j = 0; j < l.classes; j++) {
					prob.push_back(dets[i].prob[j]);
				}
				auto objectOffset = darknetServer::CreateDetectedObjectDirect(*(elem.messageBuilder), &box, dets[i].classes, dets[i].objectness, dets[i].sort_class, &prob);
				objects.push_back(objectOffset);
				numObjects++;
			}

			flatbuffers::Offset<DetectedObjects> detectedObjectsOffset = darknetServer::CreateDetectedObjectsDirect(*(elem.messageBuilder), numObjects, &objects);

			(elem.messageBuilder)->Finish(detectedObjectsOffset);
			*(elem.responseMessage) = (elem.messageBuilder)->ReleaseMessage<DetectedObjects>();
			assert((elem.responseMessage)->Verify());

			elem.done = true;
			std::cout << elem.tag << " doDetection: took " << probe_time_end2(&ts_detect) << " milliseconds"<< std::endl;

			// Clean up
			free_detections(dets, nboxes);
			//delete [] newImage.data;
		}

	private:
		void convertFrameToImage(const darknetServer::KeyFrame *frame, image *newImage) {
			newImage->w = frame->width();
			newImage->h = frame->height();
			newImage->c = frame->numChannels();
			//int dataSize = frame->data()->size();
			newImage->data =  const_cast<float *>(frame->data()->data());
			//auto dataVector = frame->data();
			//for (int i = 0; i < dataSize; i+=sizeof(float))
				//newImage->data[i] = dataVector->Get(i);
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
			axpy_cpu(this->numNetworkOutputs, 1.0, predictions, 1, average, 1);

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
		struct timestamp ts_detect;
		struct timestamp ts_gpu;
		float *predictions;
		float *average;
		char *datacfg;
		char *cfgfile;
		char *weightfile;

		network *net;
		int numNetworkOutputs;

	}; // class Detector

	// class AsyncDetector : Detector
	// {
	//   public:
	// 	void Init(int argc, char** argv, DetectionQueue *requestQueue, DetectionQueue *completionQueue) {
	// 		// Store pointers to the workQueues
	// 		this->requestQueue = requestQueue;
	// 		this->completionQueue = completionQueue;
	// 		Detector::Init(argc, argv);
	// 	}

	// 	void Shutdown() {
	// 		// Set locally owned pointers to NULL;
	// 		this->requestQueue = nullptr;
	// 		this->completionQueue = nullptr;
	// 		Detector::Shutdown();
	// 	}

	// 	void doDetection() {
	// 		while(true) {
	// 			WorkRequest elem;
	// 			// Wait on the requestQueue
	// 			requestQueue->pop_front(elem);
	// 			Detector::doDetection(elem);
	// 			// Put the result back on the completionQueue.
	// 			completionQueue->push_back(elem);
	// 		}
	// 	}

	//   private:
	// 	DetectionQueue *requestQueue;
	// 	DetectionQueue *completionQueue;

	// };

} // namespace DarknetWrapper

#endif // DARKNET_WRAPPER_CPP
