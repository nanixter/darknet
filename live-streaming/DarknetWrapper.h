#ifndef DARKNET_WRAPPER_HPP
#define DARKNET_WRAPPER_HPP

#include <thread>
#include <cstring>

#include <queue>
#include <mutex>
#include <condition_variable>

#include "utils/Timer.h"

extern "C" {
	#undef __cplusplus
	#include "darknet.h"
	#define __cplusplus 1
}

namespace DarknetWrapper {

	typedef struct
	{
		bool done;
		bool finished;
		image img;
		detection *dets;
		int nboxes;
		int classes;
		void *tag;
	} WorkRequest;


	class DetectionQueue {
	public:

		void push_back(WorkRequest &elem) {
			std::lock_guard<std::mutex> lock(this->mutex);
			this->queue.push(elem);
			this->cv.notify_one();
		}

		void push_back(std::vector<WorkRequest> &elems) {
			std::lock_guard<std::mutex> lock(this->mutex);
			for (auto elemIterator = elems.begin(); elemIterator != elems.end(); elemIterator++) {
				this->queue.push(*elemIterator);
			}
			this->cv.notify_one();
		}

		// pops 1 element
		void pop_front(WorkRequest &elem) {
			std::unique_lock<std::mutex> lock(this->mutex);
			if(this->queue.empty())
				cv.wait(lock, [this](){ return !this->queue.empty(); });

			// Once the cv wakes us up....
			if(!this->queue.empty()) {
				elem = (this->queue.front());
				this->queue.pop();
			}
		}

		// Pops upto N elements
		void pop_front(std::vector<WorkRequest> &elems, int &numElems) {
			std::unique_lock<std::mutex> lock(this->mutex);
			if(this->queue.empty())
				cv.wait(lock, [this](){ return !this->queue.empty(); });

			// Once the cv wakes us up....
			int numPopped = 0;
			while( !this->queue.empty() && (numPopped < numElems) ) {
				elems.insert(elems.end(), this->queue.front());
				this->queue.pop();
				numPopped++;
			}
			numElems = numPopped;
		}

	private:
		std::queue<WorkRequest> queue;
		std::mutex mutex;
		std::condition_variable cv;

	}; // class DetectionQueue

	class Detector {
	public:

		void Init(int argc, char** argv, int gpuNo) {
			// Initialization: Load config files, labels, graph, etc.,
			// Config the GPU and get into a thread that is ready to accept
			// images for detection.
			#ifdef GPU
			cuda_set_device(gpuNo);
			#endif
			char *cfgfile = argv[1];
			char *weightfile = argv[2];

			this->net = load_network(cfgfile, weightfile, 0);
		}

		void Shutdown() {
			// Free any darknet resources held. Close the GPU connection, etc...
			// This buffer doesn't belong to the network...
			// Don't free it. The caller provided the buffer.
			net->input_gpu = nullptr;
			free_network(this->net);
		}

		void doDetection(WorkRequest &elem) {
			timer_gpu.reset();

			float nms = .4;
			set_batch_network(net, 1);
			layer l = net->layers[net->n-1];

			network_predict_gpubuffer(net, elem.img.data);

			// This helper function can scale the boxes to the original image size.
			elem.dets = get_network_boxes(this->net, elem.img.w, elem.img.h, 0.5, 0.5, 0, 1, &(elem.nboxes));

			// Non-maximum suppression whatever that is...
			if (nms > 0) {
				do_nms_obj(elem.dets, elem.nboxes, l.classes, nms);
			}

			elem.classes = l.classes;
			//std::cout << l.classes <<std::endl;
			elem.done = true;

			std::cout << " GPU processing took " << timer_gpu.getElapsedMicroseconds() << " microseconds. numDetections =" <<elem.nboxes << std::endl;
		}

		float * getOutput() {
			layer l = get_network_output_layer(net);
			return l.output;
		}

		void getInput(float *buffer, int size) {
			cuda_pull_array(net->input_gpu, buffer, size);
		}

	private:

		// All the darknet globals.
		Timer timer_gpu;
		Timer timer_detection;

		network *net;

		float **baseOutput;

	}; // class Detector

	class QueuedDetector : Detector
	{
	public:
		void Init(int argc, char** argv, DetectionQueue *requestQueue, DetectionQueue *completionQueue, int gpuNo) {
			// Store pointers to the workQueues
			this->requestQueue = requestQueue;
			this->completionQueue = completionQueue;
			Detector::Init(argc, argv, gpuNo);
		}

		void Shutdown() {
			// Set locally owned pointers to NULL;
			this->requestQueue = nullptr;
			this->completionQueue = nullptr;
			Detector::Shutdown();
		}

		void doDetection() {
			std::vector<WorkRequest> elems;
			bool finished = false;
			elems.reserve(1);
			while(true) {
				int numImages = 1;

				// Wait on the requestQueue
				requestQueue->pop_front(elems, numImages);

				// Do the detection
				for (int i = 0; i < numImages; i++){
					// Break if this flag is set to indicate that there is no more work.
					if (elems[i].finished == true){
						this->finished == true;
						break;
					}
					Detector::doDetection(elems[i]);
				}

				// Put the result back on the completionQueue.
				completionQueue->push_back(elems);

				// Break if this flag is set to indicate that there is no more work.
				if(this->finished == true)
					break;

				// Clear the vector so we can use it again.
				elems.clear();
			}
		}

	private:
		DetectionQueue *requestQueue;
		DetectionQueue *completionQueue;
	};

} // namespace DarknetWrapper

#endif // DARKNET_WRAPPER_CPP
