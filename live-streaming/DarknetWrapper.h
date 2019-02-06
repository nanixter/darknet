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

class Detector {
public:

	void Init(int argc, char** argv, int gpuNo) {
		// Initialization: Load config files, labels, graph, etc.,
		// Config the GPU and get into a thread that is ready to accept
		// images for detection.
		#ifdef GPU
		cuda_set_device(gpuNo);
		this->gpuNum = gpuNo;
		cudaStreamCreateWithFlags(&DNNStream,cudaStreamNonBlocking);
		#endif
		char *cfgfile = argv[1];
		char *weightfile = argv[2];

		this->net = load_network(cfgfile, weightfile, 0);
		this->net->stream = &DNNStream;
	}

	void Shutdown() {
		// Free any darknet resources held. Close the GPU connection, etc...
		free_network(this->net);
	}

	void doDetection(WorkRequest &elem) {
		cuda_set_device(gpuNum);
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

	//	LOG(INFO) << " GPU processing: transfer_needed: " <<transfer <<" took " << timer_gpu.getElapsedMicroseconds()
	//				<< " microseconds. numDetections =" <<elem.nboxes;
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
	cudaStream_t DNNStream;
	network *net;

	float **baseOutput;

}; // class Detector

#endif // DARKNET_WRAPPER_CPP
