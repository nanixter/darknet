#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <thread>
#include <cmath>
#include <sys/time.h>

#include <cuda_runtime_api.h>
#include <curand.h>
#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <nppversion.h>

#include <assert.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Custom C++ Wrapper around Darknet.
#include "DarknetWrapper.h"

// Simple wrapper around NVDEC and NVENC distributed by NVIDIA
#include <NvPipe.h>

#include "nvpipe/src/NvCodec/Utils/Logger.h"
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

// Utils from NVIDIA to DEMUX and MUX video streams
#include "nvpipe/src/NvCodec/Utils/FFmpegDemuxer.h"
#include "nvpipe/src/NvCodec/Utils/FFmpegStreamer.h"

#include "utils/cudaYUV.h"
#include "utils/cudaResize.h"
#include "utils/cudaRGB.h"
#include "utils/cudaOverlay.h"

#include "utils/Timer.h"

using DarknetWrapper::WorkRequest;
// using DarknetWrapper::DetectionQueue;
using DarknetWrapper::Detector;

void printUsage(char *binaryName) {
	std::cout << "Usage:" << std::endl
			<< binaryName << " <cfg> <weights> -f <vid_file>"
			<< std::endl;
}

// Assumes the box is scaled to 416x416 image
float4 scale_box416(box bbox, int width, int height)
{
	// Convert from x, y, w, h to xleft, ytop, xright, ybottom
	//  ---------------------------------------------------------
	// |xleft,ytop-> .----------.
	// |			 |			|
	// |			 |	   .<-----------(x,y,w,h)
	// |			 |			|
	// |			 .----------. <-xright,ybottom

	//std::cout << "Scale_box416:" << bbox.x 	<<" " << bbox.y	<<" " << bbox.w <<" " << bbox.h	<< std::endl;
	float4 box;
	box.x = std::max( (bbox.x - bbox.w/2.0) * width, 0.0);
	box.y = std::max( (bbox.y - bbox.h*0.75) * height, 0.0);
	box.z = std::min( (bbox.x + bbox.w/2.0) * width, width-1.0);
	box.w = std::min( (bbox.y + bbox.h*0.75) * height, height-1.0);
	return box;
}

int main(int argc, char* argv[])
{

	if(argc < 5){
		std::cout << "Too few arguments provided." << std::endl;
		printUsage(argv[0]);
		return EXIT_FAILURE;
	}
	// Parse command-line options.
	// TODO: support RTMP ingestion (or some other network ingestion)
	char *filename;
	for(int i = 0; i < argc-1; ++i){
		if(0==strcmp(argv[i], "-f")){
			filename = argv[i+1];
		}
	}

	if (NULL == filename) {
		std::cout << "Please provide input video file." << std::endl;
		printUsage(argv[0]);
		return EXIT_FAILURE;
	}
	std::cout << "video file:" << filename <<std::endl;

	// Create the demuxer (used to read the video stream (h264/h265) from the container (mp4/mkv))
	FFmpegDemuxer demuxer(filename);

	// Formats supported by NVDEC/CUVID
	// AV_CODEC_ID_MPEG1VIDEO, AV_CODEC_ID_MPEG2VIDEO,
	// AV_CODEC_ID_H264, AV_CODEC_ID_HEVC/AV_CODEC_ID_H265,
	// AV_CODEC_ID_MJPEG, AV_CODEC_ID_MPEG4, AV_CODEC_ID_VC1,
	// AV_CODEC_ID_VP8, AV_CODEC_ID_VP9

	// NvPipe only supports H264 and HEVC, though

	NvPipe_Codec codec;
	uint32_t inWidth = demuxer.GetWidth();
	uint32_t inHeight = demuxer.GetHeight();
	uint32_t bitDepth = demuxer.GetBitDepth();
	AVRational inTimeBase = demuxer.GetTimeBase();

	std::cout << "Timebase numerator/denominator = " <<inTimeBase.num << "/" << inTimeBase.den <<std::endl;
	switch(demuxer.GetVideoCodec())	{
		case AV_CODEC_ID_H264:
			codec = NVPIPE_H264;
			break;
		case AV_CODEC_ID_H265:
			codec = NVPIPE_HEVC;
			break;
		default:
			std::cout << "Support for this video codec isn't implemented yet. NVPIPE only supports H264 and H265/HEVC" << std::endl;
			return -1;
	}

	// We're running detection on GPU 3 and Decode/Encode on GPU 4
	// Enable P2P access.
	cudaSetDevice(2);
	cudaDeviceEnablePeerAccess(3, 0);
	cudaSetDevice(3);
	cudaDeviceEnablePeerAccess(2, 0);

	// Create decoder
	NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_NV12, codec);
	if (!decoder)
		std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;

	// Encoder properties
	const float bitrateMbps = 2;
	const uint32_t targetFPS = 30;

	// Create encoder
	NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY, bitrateMbps * 1000 * 1000, targetFPS);
	if (!encoder)
		std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;

	// Create the output stream writer wrapper
	FFmpegStreamer muxer(AV_CODEC_ID_H264, inWidth, inHeight, targetFPS, inTimeBase, "./scaled.mp4");
	//FFmpegStreamer muxer(AV_CODEC_ID_H264, inWidth, inHeight, targetFPS, inTimeBase, "./scaled.mp4");

	uint8_t *compressedFrame = nullptr;
	int compressedFrameSize = 0;
	int dts = 0;
	int pts = 0;

	// NvPipe expects us to allocate a buffer for it to output to.. Sigh...
	uint8_t *compressedOutFrame = new uint8_t[200000];

	std::cout << "Initializing detector" <<std::endl;
	Detector detector;
	uint32_t netWidth = 416;
	uint32_t netHeight = 416;
	detector.Init(argc, argv, 2);

	Timer timer;
	uint64_t frameNum = 0;
	cv::Mat picRGB, picBGR;
	// In a loop, grab compressed frames from the demuxer.
	while(demuxer.Demux(&compressedFrame, &compressedFrameSize, &dts)) {
		timer.reset();
		// Allocate GPU memory and copy the compressed Frame to it
		void *compressedFrameDevice = nullptr;
		cudaMalloc(&compressedFrameDevice, compressedFrameSize);
		cudaMemcpy(compressedFrameDevice, compressedFrame, compressedFrameSize, cudaMemcpyHostToDevice);

		void *decompressedFrameDevice = nullptr;
		cudaMalloc(&decompressedFrameDevice, inWidth*inHeight*4);

		// Decode the frame
		uint64_t decompressedFrameSize = NvPipe_Decode(decoder, compressedFrame, compressedFrameSize, decompressedFrameDevice, inWidth, inHeight);
		if (decompressedFrameSize <= compressedFrameSize) {
			std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;
			exit(-1);
		}

		// Uncomment this block to dump raw frames as a sanity check.
/*		void *decompressedFrameHost = new unsigned char[decompressedFrameSize/sizeof(unsigned char)];
		cudaMemcpy(decompressedFrameHost, decompressedFrameDevice, decompressedFrameSize, cudaMemcpyDeviceToHost);

		cv::Mat picYV12 = cv::Mat(inHeight * 3/2, inWidth, CV_8UC1, decompressedFrameHost);
		cv::cvtColor(picYV12, picBGR, cv::COLOR_YUV2BGR_NV12);
		cv::imwrite("raw.bmp", picBGR);  //only for test
*/
		// Convert to RGB from NV12
		void *decompressedFrameRGBDevice = nullptr;
		cudaMalloc(&decompressedFrameRGBDevice, inWidth*inHeight*sizeof(float3));

		cudaError_t status = cudaNV12ToRGBf(static_cast<uint8_t *>(decompressedFrameDevice),
											static_cast<float3 *>(decompressedFrameRGBDevice),
											(size_t)inWidth,
											(size_t)inHeight);
		if (status != cudaSuccess)
			std::cout << "cudaNV12ToRGBf Status = " << cudaGetErrorName(status) << std::endl;
		assert(status == cudaSuccess);

		// Uncomment this block to dump raw frames as a sanity check.
/*		float3 *decompressedFrameRGBHost = new float3[inWidth*inHeight];
		cudaMemcpy((void *)decompressedFrameRGBHost, decompressedFrameRGBDevice, sizeof(float3)*inWidth*inHeight, cudaMemcpyDeviceToHost);
		std::cout << decompressedFrameRGBHost[0].x << " " << decompressedFrameRGBHost[0].y << " " <<decompressedFrameRGBHost[0].z << std::endl;
		//std::cout << "RGB data: " <<std::endl;
		//for (int i = 0; i < inWidth*inHeight; i++) {
		//	float3 temp = decompressedFrameRGBHost[i];
		//	std::cout << "Red " << temp.x
		//				<< "Green " << temp.y
		//				<< "Blue " << temp.z
		//				<< "Alpha " << temp.w
		//				<< std::endl;
		//}
		picRGB = cv::Mat(inHeight, inWidth, CV_32FC3, decompressedFrameRGBHost);
		cv::cvtColor(picRGB, picBGR, cv::COLOR_RGB2BGR);
		cv::imwrite("bgr.bmp", picBGR);  //only for test
*/
		// Scale the frame to netWidth;netHeight
		size_t noPadWidth = netWidth;
		size_t noPadHeight = netHeight;

		// Keep aspect ratio
		double h1 = netWidth * (inHeight/(double)inWidth);
		double w2 = netHeight * (inWidth/(double)inHeight);
		if( h1 <= netHeight)
			noPadHeight = (int)h1;
		else
			noPadWidth = (int)w2;

		void *scaledFrameNoPad = nullptr;
		cudaMalloc(&scaledFrameNoPad, noPadWidth*noPadHeight*sizeof(float3));

		status = cudaResizeRGB(static_cast<float3 *>(decompressedFrameRGBDevice),
								(size_t)inWidth,
								(size_t)inHeight,
								static_cast<float3 *>(scaledFrameNoPad),
								noPadWidth,
								noPadHeight);

		if (status != cudaSuccess)
			std::cout << "cudaResizeRGB Status = " << cudaGetErrorName(status) << std::endl;
		assert(status == cudaSuccess);

		// Pad the image with black border if needed
		int top = (netHeight - noPadHeight) / 2;
		int left = (netWidth - noPadWidth) / 2;

		void *scaledFramePadded = nullptr;
		cudaMalloc(&scaledFramePadded, netWidth*netHeight*sizeof(float3));

		const Npp32f bordercolor[3] = {0.0,0.0,0.0};

		NppStatus nppStatus = nppiCopyConstBorder_32f_C3R(static_cast<const Npp32f *>(scaledFrameNoPad),
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

		Npp32f *scaledPaddedPlanar[3] = {nullptr,nullptr,nullptr};
		void *scaledPaddedPlanarS = nullptr;
		cudaMalloc(&scaledPaddedPlanarS, netWidth*netHeight*sizeof(float)*3);

		//std::cout << "scaledPlanar = " << scaledPaddedPlanarS << " Last memory location = " << scaledPaddedPlanarS+netWidth*netHeight*sizeof(float)*3-1 <<std::endl;
		scaledPaddedPlanar[0] = (float *)scaledPaddedPlanarS;
		scaledPaddedPlanar[1] = (float *)(scaledPaddedPlanarS+netWidth*netHeight*sizeof(float));
		scaledPaddedPlanar[2] = (float *)(scaledPaddedPlanarS+netWidth*netHeight*sizeof(float));
		//std::cout << "scaledPlanar[0] = " << scaledPaddedPlanar[0] <<std::endl; 
		//std::cout << "scaledPlanar[1] = " << scaledPaddedPlanar[1] <<std::endl; 
		//std::cout << "scaledPlanar[2] = " << scaledPaddedPlanar[2] <<std::endl; 

		nppStatus = nppiCopy_32f_C3P3R(static_cast<const Npp32f *>(scaledFramePadded),
											netWidth*sizeof(float3),
											scaledPaddedPlanar,
											netWidth*sizeof(float),
											(NppiSize){netWidth, netHeight});

		if (nppStatus != NPP_SUCCESS)
			std::cout << "nppiCopy_32f_C3P3R Status = " << status << std::endl;
		assert(nppStatus == NPP_SUCCESS);

		float *planarRGBHost = new float[netWidth*netHeight*3];
		//cudaMemcpy((void *)planarRGBHost, scaledPaddedPlanarS, sizeof(float)*netWidth*netHeight*3, cudaMemcpyDeviceToHost);
		//cudaMemcpy((void *)scaledPaddedPlanarS, planarRGBHost, sizeof(float)*netWidth*netHeight*3, cudaMemcpyHostToDevice);
/*		picRGB = cv::Mat(netHeight, netWidth, CV_32FC3, letterboxedRGBHost);
		cv::cvtColor(picRGB, picBGR, cv::COLOR_RGB2BGR);
		cv::imwrite("letterboxed.bmp", picBGR);  //only for test
		exit(0);
*/
		image img;
		img.w = netWidth;
		img.h = netHeight;
		img.c = 3;
		img.data = (float *)scaledPaddedPlanarS;
		
		// Pass image pointer to Darknet for detection
		WorkRequest work;
		work.done = false;
		work.img = img;
		work.dets = nullptr;
		work.nboxes = 0;
		work.classes = 0;

		// The actual processing.
		detector.doDetection(work);

		// Copy detected objects to the Request
		assert(work.done == true);
		assert(work.dets != nullptr);
		
/*		detector.getInput((float*)letterboxedRGBHost, 3*netWidth*netHeight);
		std::cout << letterboxedRGBHost[top*netWidth].x << " " << letterboxedRGBHost[top*netWidth].y << " " <<letterboxedRGBHost[top*netWidth].z << std::endl;
		picRGB = cv::Mat(netHeight, netWidth, CV_32FC3, letterboxedRGBHost);
		cv::cvtColor(picRGB, picBGR, cv::COLOR_RGB2BGR);
		cv::imwrite("letterboxedInput.bmp", picBGR);  //only for test

		picRGB = cv::Mat(netHeight, netWidth, CV_32FC3, detector.getOutput());
		cv::cvtColor(picRGB, picBGR, cv::COLOR_RGB2BGR);
		cv::imwrite("letterboxedOut.bmp", picBGR);  //only for test
*/
		// Draw boxes in the decompressedFrameDevice using cudaOverlayRectLine
		int numObjects = 0;
		std::vector<float4> boundingBoxes;
		//std::cout << "nboxes = " <<work.nboxes <<std::endl; 
		for (int i = 0; i < work.nboxes; i++) {
		//	std::cout << "objectness = " << work.dets[i].objectness <<std::endl; 
			if(work.dets[i].objectness ==  0.0) continue;
//			bool draw = false;
//			for (int j = 0; j < work.classes; j++) {
//				std::cout << "prob = " << work.dets[i].prob[j] <<std::endl; 
//				if(work.dets[i].prob[j] > 0.5)
//					draw = true;
//			}
			boundingBoxes.push_back(scale_box416(work.dets[i].bbox, inWidth, inHeight));
			numObjects++;
		}

		//std::cout << "Num Objects detected = " << numObjects << std::endl;
		//for(int i = 0; i < numObjects; i++) {
		//	std::cout << "Box " << i << ": " << boundingBoxes[i].x <<" "<< boundingBoxes[i].y <<" " << boundingBoxes[i].z
		//				<<" " << boundingBoxes[i].w << std::endl;
		//}

		if (numObjects >0) {
			const float4 lineColor = {75.0, 156.0, 211.0,120.0}; 

			void *boundingBoxesDevice = nullptr;
			cudaMalloc(&boundingBoxesDevice, numObjects*sizeof(float4));
			cudaMemcpy(boundingBoxesDevice, boundingBoxes.data(), numObjects*sizeof(float4), cudaMemcpyHostToDevice);

			status = cudaRectOutlineOverlay((float3 *)decompressedFrameRGBDevice, (float3 *)decompressedFrameRGBDevice, 
											inWidth, inHeight, (float4 *)boundingBoxesDevice, numObjects, lineColor);

			if (status != cudaSuccess)
				std::cout << "cudaRectOutlineOverlay Status = " << cudaGetErrorName(status)	<< std::endl;
			assert(status == cudaSuccess);
			cudaFree(boundingBoxesDevice);
		}

		// Draw labels using cudaFont
		// Maybe later.

		// Free the detections
		free_detections(work.dets, work.nboxes);

		status = cudaRGBToBGRA8((float3 *)decompressedFrameRGBDevice, (uchar4*) decompressedFrameDevice, inWidth, inHeight);

		if (status != cudaSuccess)
			std::cout << "cudaRGBToRGBA8 Status = " << cudaGetErrorName(status)	<< std::endl;
		assert(status == cudaSuccess);

/*		uchar4 *detectedRGBHost = new uchar4[inWidth*inHeight];
		cudaMemcpy((void *)detectedRGBHost, decompressedFrameDevice, sizeof(uchar4)*inWidth*inHeight, cudaMemcpyDeviceToHost);
		picRGB = cv::Mat(inHeight, inWidth, CV_8UC4, detectedRGBHost);
		cv::cvtColor(picRGB, picBGR, cv::COLOR_BGRA2BGR);
		cv::imwrite("detected.bmp", picBGR);  //only for test
		exit(0);
*/
		// Encode the processed Frame
		uint64_t size = NvPipe_Encode(encoder, decompressedFrameDevice, inWidth*4, compressedOutFrame, 200000, inWidth, inHeight, false);
		if (0 == size)
			std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

		// MUX the frame
		muxer.Stream(compressedOutFrame, size, frameNum);
		cudaFree(compressedFrameDevice);
		cudaFree(decompressedFrameDevice);
		cudaFree(decompressedFrameRGBDevice);
		cudaFree(scaledFrameNoPad);
		cudaFree(scaledFramePadded);
		cudaFree(scaledPaddedPlanarS);

		std::cout << "Processing frame " << frameNum++ << " took " << timer.getElapsedMicroseconds() << " us." << std::endl;
	}

	NvPipe_Destroy(encoder);
	NvPipe_Destroy(decoder);

	return 0;
}
