#ifndef DRAWTHREAD_H
#define DRAWTHREAD_H

#include "utils/Types.h"
#include "utils/Queue.h"
#include "utils/PointerMap.h"

using LiveStreamDetector::Frame;
using LiveStreamDetector::WorkRequest;
using LiveStreamDetector::MutexQueue;
using LiveStreamDetector::PointerMap;

class DrawingThread {
public:
	void Init(int gpuNum, NvPipe_Codec codec,
				MutexQueue<WorkRequest> *completionQueue,
				std::vector<PointerMap<Frame> *> &detectedFrameMaps,
				int targetFPS, int inWidth, int inHeight, int numStreams)
	{
		this->completionQueue = completionQueue;
		this->detectedFrameMaps = detectedFrameMaps;
		this->targetFPS = targetFPS;
		this->inWidth = inWidth;
		this->inHeight = inHeight;
		this->gpuNum = gpuNum;
		this->numStreams = numStreams;

		cudaSetDevice(gpuNum);

		this->thread = std::thread(&DrawingThread::DrawBoxes, this);
	}

	void ShutDown()
	{
		thread.join();
	}

	float4 scale_box(box bbox, int width, int height)
	{
		// Convert from x, y, w, h to xleft, ytop, xright, ybottom
		//  ---------------------------------------------------------
		// |xleft,ytop-> .----------.
		// |			 |			|
		// |			 |	   .<-----------(x,y,w,h)
		// |			 |			|
		// |			 .----------. <-xright,ybottom

		float4 box;
		box.x = std::max( (bbox.x - bbox.w/2.0) * width, 0.0);
		box.y = std::max( (bbox.y - bbox.h*0.75) * height, 0.0);
		box.z = std::min( (bbox.x + bbox.w/2.0) * width, width-1.0);
		box.w = std::min( (bbox.y + bbox.h*0.75) * height, height-1.0);
		return box;
	}

	void DrawBoxes()
	{
		const float4 overlayColor = {75.0, 156.0, 211.0,120.0};
		while(true) {
			WorkRequest work;
			completionQueue->pop_front(work);
			// Break out of the loop if this is the completion signal.
			if(work.finished == true) {
				Frame *frame = (Frame *)work.tag;
				detectedFrameMaps[frame->streamNum]->insert(frame, frame->frameNum);
				break;
			}

			assert(work.done == true);
			assert(work.dets != nullptr);

			Frame *frame = (Frame *)work.tag;
			// Draw boxes in the decompressedFrameDevice using cudaOverlayRectLine
			int numObjects = 0;
			std::vector<float4> boundingBoxes;
			for (int i = 0; i < work.nboxes; i++) {
				if(work.dets[i].objectness ==  0.0) continue;
				// Fix this later by adding the COCO classes to the network.
	/*			bool draw = false;
				for (int j = 0; j < work.classes; j++) {
					std::cout << "prob = " << work.dets[i].prob[j] <<std::endl;
					if(work.dets[i].prob[j] > 0.5)
						draw = true;
				}
	*/
				boundingBoxes.push_back(scale_box(work.dets[i].bbox, inWidth, inHeight));
				numObjects++;
			}

			cudaError_t status;
			NppStatus nppStatus;

			if (frame->deviceNumRGB != gpuNum) {
				void *frameDevice = nullptr;
				cudaMalloc(&frameDevice, frame->decompressedFrameRGBSize);
				status = cudaMemcpyPeer(frameDevice, gpuNum, frame->decompressedFrameRGBDevice,
								frame->deviceNumRGB, frame->decompressedFrameRGBSize);
				if (status != cudaSuccess)
					std::cout << "DrawThread1 cudaMemcpyPeer Status = "<< cudaGetErrorName(status)
							<< std::endl;
				cudaFree(frame->decompressedFrameRGBDevice);
				frame->decompressedFrameRGBDevice = frameDevice;
				frame->deviceNumRGB = gpuNum;
			}

			if (numObjects >0) {
				void *boundingBoxesDevice = nullptr;
				cudaMalloc(&boundingBoxesDevice, numObjects*sizeof(float4));
				cudaMemcpy(boundingBoxesDevice, boundingBoxes.data(), numObjects*sizeof(float4), cudaMemcpyHostToDevice);

				status = cudaRectOutlineOverlay((float3 *)frame->decompressedFrameRGBDevice,
												(float3 *)frame->decompressedFrameRGBDevice,
												inWidth,
												inHeight,
												(float4 *)boundingBoxesDevice,
												numObjects,
												overlayColor);

				if (status != cudaSuccess)
					std::cout << "cudaRectOutlineOverlay Status = " << cudaGetErrorName(status)	<< std::endl;
				assert(status == cudaSuccess);
				cudaFree(boundingBoxesDevice);
			}

			// Draw labels using cudaFont
			// Maybe later.

			// Free the detections
			free_detections(work.dets, work.nboxes);

			if (frame->deviceNumDecompressed != gpuNum) {
				void *frameDevice = nullptr;
				cudaMalloc(&frameDevice, frame->decompressedFrameSize);
				status = cudaMemcpyPeer(frameDevice, gpuNum, frame->decompressedFrameDevice,
								frame->deviceNumDecompressed, frame->decompressedFrameSize);
				if (status != cudaSuccess)
					std::cout << "DrawThread2 cudaMemcpyPeer Status = "<< cudaGetErrorName(status)
							<< std::endl;
				cudaFree(frame->decompressedFrameDevice);
				frame->decompressedFrameDevice = frameDevice;
				frame->deviceNumDecompressed = gpuNum;
			}

			// Convert to uchar4 BGRA8 that the Encoder wants
			status = cudaRGBToBGRA8((float3 *)frame->decompressedFrameRGBDevice,
									(uchar4*)frame->decompressedFrameDevice,
									inWidth,
									inHeight);

			if (status != cudaSuccess)
				std::cout << "cudaRGBToRGBA8 Status = " << cudaGetErrorName(status)	<< std::endl;
			assert(status == cudaSuccess);
			cudaFree(frame->decompressedFrameRGBDevice);
			detectedFrameMaps[frame->streamNum]->insert(frame, frame->frameNum);
			pthread_yield();
		}
		LOG(INFO) <<"Draw Threds done. Exiting";
	}

private:
	// Variables we operate on
	int gpuNum;
	int targetFPS;
	int inWidth;
	int inHeight;
	int numStreams;
	std::thread thread;

	// Objects (or pointers to)
	// Create one output stream writer wrapper per thread
	MutexQueue<WorkRequest> *completionQueue;
	std::vector<PointerMap<Frame> *>  detectedFrameMaps;
};

#endif
