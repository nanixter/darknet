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

// Custom C++ Wrapper around Darknet.
//#include "DarknetWrapper.h"

// Simple wrapper around NVDEC and NVENC distributed by NVIDIA
#include <NvPipe.h>

#include "nvpipe/src/NvCodec/Utils/Logger.h"
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

// Utils from NVIDIA to DEMUX and MUX video streams
#include "nvpipe/src/NvCodec/Utils/FFmpegDemuxer.h"
#include "nvpipe/src/NvCodec/Utils/FFmpegStreamer.h"

#include "utils/cudaYUV.h"
#include "utils/cudaResize.h"

#include "utils/Timer.h"

#include "darknet.h"

int main(int argc, char* argv[])
{
	// Parse command-line options.
	// TODO: support RTMP ingestion (or some other network ingestion)
	char *filename;
	for(int i = 0; i < argc-1; ++i){
		if(0==strcmp(argv[i], "-f")){
			filename = argv[i+1];
		}
	}
	if (NULL == filename) {
		std::cout << "Usage:" << std::endl << argv[0] << " -f <vid_file>" << std::endl;
		return -1;
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

	// Create decoder
	NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_NV12, codec);
	//NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, codec);
	if (!decoder)
		std::cerr << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;

	// Encoder properties
	const float bitrateMbps = 2;
	const uint32_t targetFPS = 30;

	// Create encoder
	NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_NV12, codec, NVPIPE_LOSSY, bitrateMbps * 1000 * 1000, targetFPS);
	//NvPipe* encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY, bitrateMbps * 1000 * 1000, targetFPS);
	if (!encoder)
		std::cerr << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;
	uint32_t outWidth = 416;
	uint32_t outHeight = 416;

	// Create the output stream writer wrapper
	FFmpegStreamer muxer(AV_CODEC_ID_H264, outWidth, outHeight, targetFPS, inTimeBase, "./scaled.mp4");

	uint8_t *compressedFrame = nullptr;
	int compressedFrameSize = 0;
	int dts = 0;
	int pts = 0;

	// NvPipe expects us to allocate a buffer for it to output to.. Sigh...
	uint8_t *compressedOutFrame = new uint8_t[200000];

	Timer timer;
	uint64_t frameNum = 0;
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

		// Convert to RGB from NV12
/*		void *decompressedFrameRGBADevice = nullptr;
		cudaMalloc(&decompressedFrameRGBADevice, inWidth*inHeight*sizeof(float4));

		cudaError_t status = cudaNV12ToRGBA(static_cast<uint8_t *>(decompressedFrameDevice),
											static_cast<uchar4 *>(decompressedFrameRGBADevice),
											(size_t)inWidth,
											(size_t)inHeight);
		if (status != cudaSuccess)
			std::cout << "cudaNV12ToRGBAf Status = " << cudaGetErrorName(status) << std::endl;
		assert(status == cudaSuccess);
*/
		// Scale the frame to outWidth;outHeight
/*		size_t noPadWidth = outWidth;
		size_t noPadHeight = outHeight;

		// Keep aspect ratio
		double h1 = outWidth * (inHeight/(double)inWidth);
		double w2 = outHeight * (inWidth/(double)inHeight);
		if( h1 <= outHeight)
			noPadHeight = (int)h1;
		else
			noPadWidth = (int)w2;

		NppiInterpolationMode interploationMode = NPPI_INTER_SUPER;

		void *scaledFrameNoPad = nullptr;
		cudaMalloc(&scaledFrameNoPad, noPadWidth*noPadHeight*4);

		status = cudaResizeRGBA(static_cast<float4 *>(decompressedFrameRGBADevice),
								(size_t)inWidth,
								(size_t)inHeight,
								static_cast<float4 *>(scaledFrameNoPad),
								noPadWidth,
								noPadHeight);

		// Pad the image with black border if needed
		int top = (outHeight - noPadHeight) / 2;
		int left = (outWidth - noPadWidth) / 2;
*/
		// void *scaledFramePadded = nullptr;
		// cudaMalloc(&scaledFramePadded, dstImageSize.width*dstImageSize.height*4);

		// const Npp8u bordercolor[4] = {0,0,0,0};

		// status = nppiCopyConstBorder_8u_C3R(static_cast<const Npp8u *>(scaledFrameNoPad),
		// 									dstImageSizeNoPad.width*4,
		// 									dstImageSizeNoPad,
		// 									static_cast<Npp8u *>(scaledFramePadded),
		// 									dstImageSize.width*4,
		// 									dstImageSize,
		// 									top,
		// 									left,
		// 									bordercolor);

		// if (status != NPP_SUCCESS)
		// 	std::cout << "NPPCopyConstBorder Status = " << status << std::endl;
		// assert(status == NPP_SUCCESS);

		// Pass image pointer to Darknet for detection

		// Encode the processed Frame
		//uint64_t size = NvPipe_Encode(encoder, scaledFrameNoPad, noPadWidth, compressedOutFrame, 200000, noPadWidth, noPadHeight, false);
		//uint64_t size = NvPipe_Encode(encoder, scaledFramePadded, outWidth * 4, compressedOutFrame, 200000, outWidth, outHeight, false);
		//uint64_t size = NvPipe_Encode(encoder, decompressedFrameDevice, inWidth * 4, compressedOutFrame, 200000, inWidth, inHeight, false);
		//uint64_t size = NvPipe_Encode(encoder, decompressedFrameRGBADevice, inWidth * 4, compressedOutFrame, 200000, inWidth, inHeight, false);
		uint64_t size = NvPipe_Encode(encoder, decompressedFrameDevice, decompressedFrameSize/inHeight, compressedOutFrame, 200000, inWidth, inHeight, false);
		if (0 == size)
			std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

		// MUX the frame
		muxer.Stream(compressedOutFrame, size, frameNum);
		cudaFree(compressedFrameDevice);
		cudaFree(decompressedFrameDevice);
		//cudaFree(decompressedFrameRGBADevice);
		//cudaFree(scaledFrameNoPad);
		// cudaFree(scaledFramePadded);

		std::cout << "Processing frame " << frameNum++ << " took " << timer.getElapsedMicroseconds() << " us." << std::endl;
	}

	NvPipe_Destroy(encoder);
	NvPipe_Destroy(decoder);

	return 0;
}
