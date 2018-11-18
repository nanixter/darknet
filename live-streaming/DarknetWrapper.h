#ifndef DARKNET_WRAPPER_HPP
#define DARKNET_WRAPPER_HPP

#include <thread>
#include <cstring>

#include <queue>
#include <mutex>
#include <condition_variable>

extern "C" {
	#undef __cplusplus
	#include "darknet.h"
	#define __cplusplus 1
}

#include "utils/Timer.h"
#include "utils/Types.h"
#include "utils/Queue.h"
#include "utils/PointerMap.h"

using LiveStreamDetector::Frame;
using LiveStreamDetector::WorkRequest;
using LiveStreamDetector::MutexQueue;
using LiveStreamDetector::PointerMap;

namespace DarknetWrapper {

	class Detector {
	public:

		void Init(int argc, char** argv, int gpuNo) {
			// Initialization: Load config files, labels, graph, etc.,
			// Config the GPU and get into a thread that is ready to accept
			// images for detection.
			#ifdef GPU
			cuda_set_device(gpuNo);
			this->gpuNum = gpuNo;
			#endif
			char *cfgfile = argv[1];
			char *weightfile = argv[2];

			this->net = load_network(cfgfile, weightfile, 0);
		}

		void Shutdown() {
			// Free any darknet resources held. Close the GPU connection, etc...
			free_network(this->net);
		}

		void doDetection(WorkRequest &elem) {
			timer_gpu.reset();

			float nms = .4;
			set_batch_network(net, 1);
			layer l = net->layers[net->n-1];

			network_predict_gpubuffer(net, elem.img.data, elem.deviceNum);

			bool transfer = (net->gpu_index == elem.deviceNum);

			// This helper function can scale the boxes to the original image size.
			elem.dets = get_network_boxes(this->net, elem.img.w, elem.img.h, 0.5, 0.5, 0, 1, &(elem.nboxes));

			// Non-maximum suppression whatever that is...
			if (nms > 0) {
				do_nms_obj(elem.dets, elem.nboxes, l.classes, nms);
			}

			elem.classes = l.classes;
			//std::cout << l.classes <<std::endl;
			elem.done = true;

			LOG(INFO) << " GPU processing: transfer_needed: " <<transfer <<" took " << timer_gpu.getElapsedMicroseconds() 
						<< " microseconds. numDetections =" <<elem.nboxes;
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
		int gpuNum;

		network *net;

		float **baseOutput;

	}; // class Detector

	class QueuedDetector : Detector
	{
	public:
		void Init(int argc, char** argv, MutexQueue<WorkRequest> *requestQueue,
					MutexQueue<WorkRequest> *completionQueue, int gpuNo) {
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
						finished == true;
						break;
					}
					Detector::doDetection(elems[i]);
				}


				// Put the result back on the completionQueue.
				completionQueue->push_back(elems);

				// Break if this flag is set to indicate that there is no more work.
				// Also insert 12 finished workRequests (12 encoder threads)
				if(finished == true){
					WorkRequest work;
					work.finished = true;
					for (int i = 0; i < 12; i++)
						completionQueue->push_back(work);
					break;
				}

				// Clear the vector so we can use it again.
				elems.clear();
				pthread_yield();
			}
		}

	private:
		MutexQueue<WorkRequest> *requestQueue;
		MutexQueue<WorkRequest> *completionQueue;
	};

} // namespace DarknetWrapper

#endif // DARKNET_WRAPPER_CPP
