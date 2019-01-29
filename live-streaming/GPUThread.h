#ifndef GPUTHREAD_H
#define GPUTHREAD_H

#include "utils/Types.h"
#include "utils/Queue.h"
#include <atomic>

using LiveStreamDetector::Frame;
using LiveStreamDetector::WorkRequest;

#include "DarknetWrapper.h"

class GPUThread {
public:
	void Init(NvPipe_Codec codec, MutexQueue<Frame> *frames,
			std::vector<PointerMap<Frame> *> &completedFramesMap,
			int firstGPU, int detectorGPU,
			int targetFPS, int inWidth, int inHeight, int numStreams,
			int argc, char** argv)
	{
		this->completedFramesMap = completedFramesMap;
		this->frames = frames;
		this->targetFPS = targetFPS;
		this->inWidth = inWidth;
		this->inHeight = inHeight;
		this->gpuNum = firstGPU;
		this->threadID = firstGPU;
		this->detectorGPU = detectorGPU;
		this->numStreams = numStreams;
		this->done.store(false, std::memory_order_release);

		// Initialize Darknet Detector
		detector.Init(argc, argv, detectorGPU);

		// Launch GPUThread
		this->thread = std::thread(&GPUThread::doGPUWork, this);
	}

	void ShutDown()
	{
		// Ensure we aren't modifying the variable while our pal is also checking it...
		this->done.store(true, std::memory_order_release);
		frames->setDone();
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

	void doGPUWork()
	{
		void *scaledFrameNoPad = nullptr;
		void *scaledFramePadded = nullptr;
		void *scaledPaddedPlanarS = nullptr;

		int noPadWidth = netWidth;
		int noPadHeight = netHeight;

		// Keep aspect ratio
		double h1 = netWidth * (inHeight/(double)inWidth);
		double w2 = netHeight * (inWidth/(double)inHeight);
		if( h1 <= netHeight)
			noPadHeight = (int)h1;
		else
			noPadWidth = (int)w2;

		int top = (netHeight - noPadHeight) / 2;
		int left = (netWidth - noPadWidth) / 2;

		const Npp32f bordercolor[3] = {0.0,0.0,0.0};

		Npp32f *scaledPaddedPlanar[3] = {nullptr,nullptr,nullptr};

		cudaMalloc(&scaledFrameNoPad, noPadWidth*noPadHeight*sizeof(float3));
		cudaMalloc(&scaledFramePadded, netWidth*netHeight*sizeof(float3));
		cudaMalloc(&scaledPaddedPlanarS, netWidth*netHeight*sizeof(float)*3);

		scaledPaddedPlanar[0] = (float *)scaledPaddedPlanarS;
		scaledPaddedPlanar[1] = ((float *)scaledPaddedPlanarS)+netWidth*netHeight;
		scaledPaddedPlanar[2] = ((float *)scaledPaddedPlanarS)+netWidth*netHeight;

		const float4 overlayColor = {75.0, 156.0, 211.0,120.0};
		cudaError_t status;
		NppStatus nppStatus;

		cudaSetDevice(gpuNum);
		cudaStream_t RDstream;
		cudaStreamCreateWithFlags(&RDstream,cudaStreamNonBlocking);

		int maxNumObjects = 20;
		void *boundingBoxesDevice = nullptr;
		cudaMalloc(&boundingBoxesDevice, maxNumObjects*sizeof(float4));

		int decompressedFrameSize = inWidth*inHeight*4;
		void *frameDevice = nullptr;
		cudaMalloc(&frameDevice, decompressedFrameSize);

		int decompressedFrameRGBSize = inWidth*inHeight*sizeof(float3);
		void *decompressedFrameRGBDevice = nullptr;
		cudaMalloc(&decompressedFrameRGBDevice, inWidth*inHeight*sizeof(float3));

		while(true) {
			Frame *frame = new Frame;
			while (frames->pop_front(*frame) == false) {
				// Check whether we've been signalled to shut down.
				if (this->done.load(std::memory_order_acquire) == true) {
					detector.Shutdown();
					goto cleanup;
				}
			}

			cudaSetDevice(gpuNum);

			bool copyBackNeeded = false;
			void *localDecompressedFrameDevice = nullptr;
			if (frame->deviceNumDecompressed != gpuNum) {
				copyBackNeeded = true;
				status = cudaMemcpyPeer(frameDevice, gpuNum,
										frame->decompressedFrameDevice,
										frame->deviceNumDecompressed,
										frame->decompressedFrameSize);
				if (status != cudaSuccess)
					std::cout << "GPUThread cudaMemcpyPeer Status = "<< cudaGetErrorName(status)
							<< std::endl;
				localDecompressedFrameDevice = frameDevice;
			} else {
				localDecompressedFrameDevice = frame->decompressedFrameDevice;
			}

			// Convert to RGB from NV12
			status = cudaNV12ToRGBf(
						static_cast<uint8_t *>(localDecompressedFrameDevice),
						static_cast<float3 *>(decompressedFrameRGBDevice),
						(size_t)inWidth,
						(size_t)inHeight,
						&RDstream);
			if (status != cudaSuccess)
				LOG(ERROR) << "cudaNV12ToRGBf Status = "<< cudaGetErrorName(status);
			assert(status == cudaSuccess);

			// Scale the frame to noPadWidth;noPadHeight
			status = cudaResizeRGB(
						static_cast<float3 *>(decompressedFrameRGBDevice),
						(size_t)inWidth,
						(size_t)inHeight,
						static_cast<float3 *>(scaledFrameNoPad),
						noPadWidth,
						noPadHeight,
						&RDstream);

			if (status != cudaSuccess)
				std::cout << "cudaResizeRGB Status = " << cudaGetErrorName(status) << std::endl;
			assert(status == cudaSuccess);

			if (nppGetStream() != RDstream)
				nppSetStream(RDstream);

			// Pad the image with black border if needed
			nppStatus = nppiCopyConstBorder_32f_C3R(
							static_cast<const Npp32f *>(scaledFrameNoPad),
							noPadWidth*sizeof(float3),
							(NppiSize){noPadWidth, noPadHeight},
							static_cast<Npp32f *>(scaledFramePadded),
							netWidth*sizeof(float3),
							(NppiSize){netWidth, netHeight},
							top,
							left,
							bordercolor);

			if (nppStatus != NPP_SUCCESS)
				std::cout << "NPPCopyConstBorder Status = " << status << std::endl;
			assert(nppStatus == NPP_SUCCESS);

			// Convert from Packed to Planar RGB (Darknet operates on planar RGB)
			nppStatus = nppiCopy_32f_C3P3R(
							static_cast<const Npp32f *>(scaledFramePadded),
							netWidth*sizeof(float3),
							scaledPaddedPlanar,
							netWidth*sizeof(float),
							(NppiSize){netWidth, netHeight});

			if (nppStatus != NPP_SUCCESS)
				std::cout << "nppiCopy_32f_C3P3R Status = " << status << std::endl;
			assert(nppStatus == NPP_SUCCESS);

			cudaStreamSynchronize(RDstream);
			// Copy image data into image struct
			image img;
			img.w = netWidth;
			img.h = netHeight;
			img.c = 3;
			img.data = (float *)scaledPaddedPlanarS;

			// Make a WorkRequest. Used to manage async execution.
			WorkRequest work;
			work.done = false;
			work.img = img;
			work.dets = nullptr;
			work.nboxes = 0;
			work.classes = 0;
			work.deviceNum = gpuNum;
			work.tag = frame;

			detector.doDetection(work);

			assert(work.done == true);
			assert(work.dets != nullptr);

			// Draw boxes using cudaOverlayRectLine
			int numObjects = 0;
			std::vector<float4> boundingBoxes;
			for (int i = 0; i < work.nboxes; i++) {
				if (numObjects == maxNumObjects) break;
				if (work.dets[i].objectness ==  0.0) continue;
				boundingBoxes.push_back(scale_box(work.dets[i].bbox, inWidth, inHeight));
				numObjects++;
			}

			cudaSetDevice(gpuNum);
			// if (frame->deviceNumRGB != gpuNum) {
			// 	void *frameDevice = nullptr;
			// 	cudaMalloc(&frameDevice, frame->decompressedFrameRGBSize);
			// 	status = cudaMemcpyPeer(frameDevice, gpuNum, decompressedFrameRGBDevice,
			// 					frame->deviceNumRGB, decompressedFrameRGBSize);
			// 	if (status != cudaSuccess)
			// 		std::cout << "DrawThread1 cudaMemcpyPeer Status = "<< cudaGetErrorName(status)
			// 				<< std::endl;
			// 	cudaFree(decompressedFrameRGBDevice);
			// 	decompressedFrameRGBDevice = frameDevice;
			// 	frame->deviceNumRGB = gpuNum;
			// }

			if (numObjects >0) {
				cudaMemcpy(boundingBoxesDevice, boundingBoxes.data(), numObjects*sizeof(float4), cudaMemcpyHostToDevice);
				status = cudaRectOutlineOverlay((float3 *)decompressedFrameRGBDevice,
												(float3 *)decompressedFrameRGBDevice,
												inWidth,
												inHeight,
												(float4 *)boundingBoxesDevice,
												numObjects,
												overlayColor,
												&RDstream);
				if (status != cudaSuccess)
					std::cout << "cudaRectOutlineOverlay Status = " << cudaGetErrorName(status)	<< std::endl;
				assert(status == cudaSuccess);
			}

			// Free the detections
			free_detections(work.dets, work.nboxes);

			// if (frame->deviceNumDecompressed != gpuNum) {
			// 	void *frameDevice = nullptr;
			// 	cudaMalloc(&frameDevice, frame->decompressedFrameSize);
			// 	status = cudaMemcpyPeer(frameDevice, gpuNum, frame->decompressedFrameDevice,
			// 					frame->deviceNumDecompressed, frame->decompressedFrameSize);
			// 	if (status != cudaSuccess)
			// 		std::cout << "DrawThread2 cudaMemcpyPeer Status = "<< cudaGetErrorName(status)
			// 				<< std::endl;
			// 	cudaFree(frame->decompressedFrameDevice);
			// 	frame->decompressedFrameDevice = frameDevice;
			// 	frame->deviceNumDecompressed = gpuNum;
			// }

			// Convert to uchar4 BGRA8 that the Encoder wants
			status = cudaRGBToBGRA8((float3 *) decompressedFrameRGBDevice,
									(uchar4*) localDecompressedFrameDevice,
									inWidth,
									inHeight,
									&RDstream);
			if (status != cudaSuccess)
				std::cout << "cudaRGBToRGBA8 Status = "<< cudaGetErrorName(status)	<< std::endl;
			assert(status == cudaSuccess);

			if (copyBackNeeded) {
				status = cudaMemcpyPeer(frame->decompressedFrameDevice,
										frame->deviceNumDecompressed,
										localDecompressedFrameDevice,
										gpuNum,
										frame->decompressedFrameSize);
				if (status != cudaSuccess)
					std::cout << "GPUThread Copy Back cudaMemcpyPeer Status = "
								<< cudaGetErrorName(status) << std::endl;
			}

			completedFramesMap[frame->streamNum]->insert(frame, frame->frameNum);

		}
	cleanup:
		LOG(INFO) << "GPUThread is done. Freeing memory and returning";
		cudaFree(scaledFrameNoPad);
		cudaFree(scaledFramePadded);
		cudaFree(scaledPaddedPlanarS);
		cudaFree(decompressedFrameRGBDevice);
		cudaFree(frameDevice);
		cudaFree(boundingBoxesDevice);
	}

private:
	int inWidth;
	int inHeight;

	int netWidth = 416;
	int netHeight = 416;

	int targetFPS;
	int gpuNum;
	int detectorGPU;
	int numStreams;
	int threadID;

	// Objects (or pointers to)
	MutexQueue<Frame> *frames;
	std::vector<PointerMap<Frame> *> completedFramesMap;
	std::atomic_bool done;
	Detector detector;
	std::thread thread;
};

#endif
