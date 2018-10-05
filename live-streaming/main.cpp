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
#include <sys/time.h>

#include <cuda_runtime_api.h>

// Custom C++ Wrapper around Darknet.
#include "DarknetWrapper.h"

// Simple wrapper around NVDEC and NVENC distributed by NVIDIA
#include <NvPipe.h>

// Utils from NVIDIA to DEMUX and MUX video streams
#include "NvCodec/Utils/FFmpegDemuxer.h"
#include "NvCodec/Utils/FFmpegStreamer.h"

#include "Timer.h"

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

	const NvPipe_Codec codec;
	uint32_t width = demuxer.GetWidth();
	uint32_t height = demuxer.GetHeight();
	uint32_t bitDepth = demuxer.GetBitDepth();
	switch(demuxer.GetVideoCodec())	{
		case AV_CODEC_ID_H264:
			codec = NVPIPE_H264;
			break;
		case AV_CODEC_ID_HEVC:
		case AV_CODEC_ID_H265:
			codec = NVPIPE_HEVC;
			break;
		default:
			std::cout << "Support for this video codec isn't implemented yet. NVPIPE only supports H264 and H265/HEVC" << std::endl;
			return -1;
	}

	// Create decoder
	NvPipe* decoder = NvPipe_CreateDecoder(NVPIPE_RGBA32, codec);
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
	FFmpegStreamer muxer(AV_CODEC_ID_H264, width, height, targetFPS, "./scaled.mp4")

	uint8_t *compressedFrame = nullptr;
	int compressedFrameSize = 0;

	Timer timer;
	// In a loop, grab compressed frames from the demuxer.
	while(demuxer.Demux(&compressedFrame, &compressedFrameSize)) {
		timer.reset();
		// Allocate GPU memory and copy the compressed Frame to it
		void* compressedFrameDevice;
		cudaMalloc(&compressedFrameDevice, compressedFrameSize);
		cudaMemcpy(compressedFrameDevice, compressedFrame, compressedFrameSize, cudaMemcpyHostToDevice);

		void* decompressedFrameDevice;
		cudaMalloc(&decompressedFrameDevice, width*height*4);

		// Decode the frame
		uint64_t ret = NvPipe_Decode(decoder, compressedFrame, compressedFrameSize, decompressedFrameDevice, width, height);
		if (ret == compressedFrameSize)
			std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;

		// Encode the processed Frame
		uint64_t size = NvPipe_Encode(encoder, decompressedFrameDevice, width * 4, compressedFrameDevice, compressedFrameSize, width, height, false);
		if (0 == size)
			std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

		cudaMemcpy(compressedFrame, compressedFrameDevice, size, cudaMemcpyDeviceToHost);
		// MUX the frame
		muxer.Stream(compressedFrame, size, demuxer.GetPTS());
		std::cout << "Frame took " << timer.getElapsedMilliseconds() << "ms." <<std::endl ;
		cudaFree(compressedFrameDevice);
		cudaFree(decompressedFrameDevice);
	}

	NvPipe_Destroy(encoder);
	NvPipe_Destroy(decoder);

	return 0;
}
