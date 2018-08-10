#ifndef DARKNET_WRAPPER_HPP
#define DARKNET_WRAPPER_HPP

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

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
		darknetServer::DetectedObjects objects;
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

		void Init(DetectionQueue *requestQueue, DetectionQueue *completionQueue) {
			// Store pointers to the workQueues
			this.requestQueue = requestQueue;
			this.completionQueue = completionQueue;

			// Initialization: Load config files, labels, graph, etc.,
			// Config the GPU and get into a thread that is ready to accept
			// images for detection.
		}

		void doDetection() {
			// wait on the requestQueue
			// Once woken up, do the actual detection
			// Put the result back on the completionQueue and call notify.
		}

		void ShutDown() {
			// Set locally owned pointers to NULL;
			requestQueue = nullptr;
			completionQueue = nullptr;

			// Free any darknet resources held. Close the GPU connection, etc...
		}

	private:
		// All the darknet globals.
		DetectionQueue *requestQueue;
		DetectionQueue *completionQueue;

	}; // class Detector

} // namespace DarknetWrapper

#endif // DARKNET_WRAPPER_CPP