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

		void doDetection(std::vector<WorkRequest> &elems, int numImages) {

			std::cout << "doDetection: new batch request. Batch Size = " << numImages << std::endl;
			timer_detection.reset();

			// Save the address of the l.output for YOLO layers so we can restore it later.
			// What a dirty hack...
			this->saveBaseOutput();

			float nms = .4;
			set_batch_network(net, numImages);
			layer l = net->layers[net->n-1];

			int bufferSize = 0;

			for (int elemNum = 0; elemNum < numImages; elemNum++)
				bufferSize += net->h*net->w*elems[elemNum].img.c;

			void *dataToProcess = nullptr;
			cudaMalloc(&dataToProcess, bufferSize*sizeof(float));


			// Copy all the images into 1 buffer.
			for (int elemNum = 0 ; elemNum < numImages; elemNum++) {
				int imgSize = net->h*net->w*elems[elemNum].img.c;
				cudaMemcpy(dataToProcess+elemNum*imgSize, elems[elemNum].img.data, imgSize*sizeof(float), cudaMemcpyDeviceToDevice);
			}

			 // Now we finally run the actual network
			timer_gpu.reset();

			network_predict_gpubuffer(net, (float *)dataToProcess);

			// Copy the detected boxes into the appropriate WorkRequest
			for (int elemNum = 0 ; elemNum < numImages; elemNum++) {
				elems[elemNum].dets = get_network_boxes(this->net, elems[elemNum].img.w, elems[elemNum].img.h,0.5, 0.5, 0, 1, &(elems[elemNum].nboxes));
				// What the hell does this do?
				if (nms > 0) {
					do_nms_obj(elems[elemNum].dets, elems[elemNum].nboxes, l.classes, nms);
				}
				// Darknet batching is kinda broken.
				// Gotta do this nonsense to set the l.output to the right address
				shiftOutput();
				elems[elemNum].classes = l.classes;
				elems[elemNum].done = true;
			}

			restoreOutputAddr();
			cudaFree(dataToProcess);

			std::cout << "Batch GPU processing took " << timer_gpu.getElapsedMicroseconds() << " milliseconds"<< std::endl;
			std::cout << " doDetection: took " << timer_detection.getElapsedMicroseconds() << " milliseconds"<< std::endl;
		}

		float * getOutput() {
			layer l = get_network_output_layer(net);
			return l.output;
		}

		void getInput(float *buffer, int size) {
			cuda_pull_array(net->input_gpu, buffer, size);
		}

	private:

		// Helper functions from https://gist.github.com/ElPatou/706a6ff36b2dce1f492007e87bcd2a0c
		void saveBaseOutput() {
			int num = 0;
			for(int i = 0; i < net->n; ++i) {
				layer *l = &(net->layers[i]);
				if (l->type == YOLO) {
					num++;
				}
			}

			baseOutput = (float **)calloc(num, sizeof(float **));

			int k = 0;
			for(int i = 0; i < net->n; ++i) {
				layer *l = &(net->layers[i]);
				if (l->type == YOLO) {
					baseOutput[k] = l->output;
					k++;
				}
			}
		}

		void restoreOutputAddr() {
			int k = 0;
			for(int i = 0; i < net->n; ++i) {
				layer *l = &(net->layers[i]);
				if (l->type == YOLO) {
					l->output = baseOutput[k];
					k++;
				}
			}
		}

		void shiftOutput() {
			for(int i = 0; i < net->n; ++i) {
				layer *l = &(net->layers[i]);
				if (l->type == YOLO) {
					l->output += l->outputs;
				}
			}
		}

		// All the darknet globals.
		Timer timer_gpu;
		Timer timer_detection;

		network *net;

		float **baseOutput;

	}; // class Detector

	class AsyncDetector : Detector
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
			elems.reserve(4);
			while(true) {
				int numImages = 4;

				// Wait on the requestQueue
				requestQueue->pop_front(elems, numImages);

				// Do the detection
				// For N=1, and N=2, just fall back to the 1 image at a time step.
				if (numImages == 1) {
					Detector::doDetection(elems[0]);
				} else if (numImages == 2) {
					Detector::doDetection(elems[0]);
					Detector::doDetection(elems[1]);
				} else{
					Detector::doDetection(elems, numImages);
				}

				// Put the result back on the completionQueue.
				completionQueue->push_back(elems);

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
