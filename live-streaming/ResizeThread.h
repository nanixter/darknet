#ifndef RESIZETHREAD_H
#define RESIZETHREAD_H

#include "utils/Types.h"
#include "utils/Queue.h"

using LiveStreamDetector::Frame;
using LiveStreamDetector::Frame;
using LiveStreamDetector::WorkRequest;

class ResizeThread {
public:
	void Init(int gpuNum, NvPipe_Codec codec, MutexQueue<Frame> *frames,
				MutexQueue<WorkRequest> *requestQueue,
				int targetFPS, int inWidth, int inHeight)
	{
		this->requestQueue = requestQueue;
		this->frames = frames;
		this->targetFPS = targetFPS;
		this->inWidth = inWidth;
		this->inHeight = inHeight;
		this->gpuNum = gpuNum;

		cudaSetDevice(gpuNum);

		// Launch ResizeThread
		this->thread = std::thread(&ResizeThread::Resize, this);
	}

	void Resize()
	{
		void *scaledFrameNoPad = nullptr;
		void *scaledFramePadded = nullptr;
		void *scaledPaddedPlanarS = nullptr;

		size_t noPadWidth = netWidth;
		size_t noPadHeight = netHeight;

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
		scaledPaddedPlanar[1] = (float *)(scaledPaddedPlanarS+netWidth*netHeight*sizeof(float));
		scaledPaddedPlanar[2] = (float *)(scaledPaddedPlanarS+netWidth*netHeight*sizeof(float));

		while(true) {
			Frame *frame = new Frame;
			frames->pop_front(*frame);
			// Check if this is a frame that was inserted to indicate that
			// there is no more work.
			if (frame->finished == true) {
				// Add a completion WorkRequest to the completion queue to signal to the
				// completion thread to finish.
				WorkRequest work;
				work.done = false;
				work.finished = true;
				work.dets = nullptr;
				work.nboxes = 0;
				work.classes = 0;
				work.tag = nullptr;
				requestQueue->push_back(work);
				break;
			}

			cudaError_t status;
			NppStatus nppStatus;

			if (frame->deviceNumDecompressed != gpuNum) {
				void *frameDevice = nullptr;
				cudaMalloc(&frameDevice, frame->decompressedFrameSize);
				cudaMemcpyPeer(frameDevice, gpuNum, frame->decompressedFrameDevice,
								frame->deviceNumDecompressed, frame->decompressedFrameSize);
				cudaFree(frame->decompressedFrameDevice);
				frame->decompressedFrameDevice = frameDevice;
				frame->deviceNumDecompressed = gpuNum;
			}

			cudaMalloc(&frame->decompressedFrameRGBDevice, inWidth*inHeight*sizeof(float3));
			frame->decompressedFrameRGBSize = inWidth*inHeight*sizeof(float3);
			frame->deviceNumRGB = gpuNum;

			// Convert to RGB from NV12
			status = cudaNV12ToRGBf(static_cast<uint8_t *>(frame->decompressedFrameDevice),
												static_cast<float3 *>(frame->decompressedFrameRGBDevice),
												(size_t)inWidth,
												(size_t)inHeight);
			if (status != cudaSuccess)
				LOG(ERROR) << "cudaNV12ToRGBf Status = " << cudaGetErrorName(status);
			assert(status == cudaSuccess);

			// Scale the frame to noPadWidth;noPadHeight
			status = cudaResizeRGB(static_cast<float3 *>(frame->decompressedFrameRGBDevice),
									(size_t)inWidth,
									(size_t)inHeight,
									static_cast<float3 *>(scaledFrameNoPad),
									noPadWidth,
									noPadHeight);

			if (status != cudaSuccess)
				std::cout << "cudaResizeRGB Status = " << cudaGetErrorName(status) << std::endl;
			assert(status == cudaSuccess);

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
			nppStatus = nppiCopy_32f_C3P3R(static_cast<const Npp32f *>(scaledFramePadded),
												netWidth*sizeof(float3),
												scaledPaddedPlanar,
												netWidth*sizeof(float),
												(NppiSize){netWidth, netHeight});

			if (nppStatus != NPP_SUCCESS)
				std::cout << "nppiCopy_32f_C3P3R Status = " << status << std::endl;
			assert(nppStatus == NPP_SUCCESS);

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
			work.finished = false;
			work.deviceNum = gpuNum;
			work.tag = frame;

			// Put packet in processing queue
			requestQueue->push_back(work);

			// Account for the time spent processing the packet...
			usleep((1000000/targetFPS));//- frame->timer.getElapsedMicroseconds());
		} // while(true)
		LOG(INFO) << "Decoder is done. Freeing memory and returning";
		cudaFree(scaledFrameNoPad);
		cudaFree(scaledFramePadded);
		cudaFree(scaledPaddedPlanarS);
	}

private:
	int inWidth;
	int inHeight;

	int netWidth = 416;
	int netHeight = 416;

	int targetFPS;
	int gpuNum;

	// Objects (or pointers to)
	MutexQueue<Frame> *frames;
	MutexQueue<WorkRequest> *requestQueue;
	std::thread thread;
};

#endif
