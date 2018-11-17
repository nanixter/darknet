#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <deque>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <thread>
#include <cmath>
#include <sys/time.h>

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <npp.h>
#include <nppi.h>
#include <npps.h>
#include <nppversion.h>

#include <assert.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// Simple wrapper around NVDEC and NVENC distributed by NVIDIA
#include <NvPipe.h>

// Gotta createt the logger before including FFmpegStreamer/Demuxer
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

// Custom C++ Wrapper around Darknet.
#include "DarknetWrapper.h"
#include "utils/PointerMap.h"
#include "utils/Types.h"
#include "utils/Queue.h"

using LiveStreamDetector::Frame;
using LiveStreamDetector::WorkRequest;
using LiveStreamDetector::MutexQueue;
using DarknetWrapper::QueuedDetector;

#include "DrawThread.h"
#include "ResizeThread.h"

void decodeFrame(NvPipe* decoder, MutexQueue<Frame> *inFrames, MutexQueue<Frame> *outFrames,
				int inWidth, int inHeight, int fps, int gpuNum, uint64_t lastFrameNum)
{
	uint64_t frameNum = 1;
	cudaSetDevice(gpuNum);

	while( frameNum < lastFrameNum ) {
		Frame frame;
		inFrames->pop_front(frame);
		frame.timer.reset();
		frame.streamNum = gpuNum;
		frame.decompressedFrameSize = inWidth*inHeight*4;
		frame.deviceNumDecompressed = gpuNum;

		cudaMalloc(&frame.decompressedFrameDevice, frame.decompressedFrameSize);

		// Decode the frame
		uint64_t decompressedFrameSize = NvPipe_Decode(decoder, (const uint8_t *)frame.data,
														frame.frameSize,
														frame.decompressedFrameDevice,
														inWidth, inHeight);

		if (decompressedFrameSize <= frame.frameSize) {
			std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;
			exit(-1);
		}

		outFrames->push_back(frame);
		usleep(1000000.0/fps);
	}
}

void encodeFrame(NvPipe *encoder, PointerMap<Frame> *inFrames, PointerMap<Frame> *outFrames,
				int inWidth, int inHeight,  int gpuNum, uint64_t lastFrameNum)
{
	uint64_t frameNum = 1;
	cudaSetDevice(gpuNum);

	while( frameNum < lastFrameNum ) {
		Frame *frame = new Frame;
		bool gotFrame = false;
		while(!gotFrame)
			gotFrame = inFrames->getElem(&frame, frameNum);

		void *frameDevice = nullptr;
		if (frame->deviceNumDecompressed != gpuNum) {
			cudaMalloc(&frameDevice, frame->decompressedFrameSize);
			cudaMemcpyPeer(frameDevice, gpuNum, frame->decompressedFrameDevice,
							frame->deviceNumDecompressed, frame->decompressedFrameSize);
			cudaFree(frame->decompressedFrameDevice);
		} else{
			frameDevice = frame->decompressedFrameDevice;
		}

		// NvPipe expects us to allocate a buffer for it to output to.. Sigh...
		delete [] frame->data;
		frame->frameSize = 500000;
		frame->data = new uint8_t[frame->frameSize];

		// Encode the processed Frame
		uint64_t size = NvPipe_Encode(encoder, frameDevice, inWidth*4, frame->data,
					frame->frameSize, inWidth, inHeight, false);

		if (0 == size)
			std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

		// Insert the encoded frame into map for the main thread to mux.
		outFrames->insert(frame, frameNum);
		cudaFree(frameDevice);
	}
}

void printUsage(char *binaryName) {
	LOG(ERROR) << "Usage:" << std::endl
			<< binaryName << " <cfg_file> <weights_file> -v <vid_file> <Opt Args>" <<std::endl
			<< "Optional Arguments:" <<std::endl
			<<	"-n number-of-clients (default=1; valid range: 1 to 12)" <<std::endl
			<<	"-s number of video streams (default=1; valid range: 1 to 4)" <<std::endl
			<<	"-f fps (default=30fps; valid range: 1 to 120)" <<std::endl
			<<	"-r per_client_max_outstanding_requests (default=90; valid range = 1 to 1000)" <<std::endl
			<<	"-b bit rate of output video (in Mbps; default=2; valid range = 1 to 6;)" <<std::endl;
}

int main(int argc, char* argv[])
{
	if (argc < 5 || 0==(argc%2)) {
		printUsage(argv[0]);
		return EXIT_FAILURE;
	}

	// Parse command-line options.
	// TODO: support RTMP ingestion (or some other network ingestion)
	char *filename;
	int numThreads = 1;
	int numStreams = 1;
	int fps = 30;
	int maxOutstandingPerThread = 90;
	float bitrateMbps = 2;

	int numPhysicalGPUs;
	cudaError_t status = cudaGetDeviceCount(&numPhysicalGPUs);
	if (status != cudaSuccess)
	std::cout << "cudaGetDeviceCount Status = " << cudaGetErrorName(status)	<< std::endl;
	assert(status == cudaSuccess);

	for (int i = 1; i < argc-1; i=i+2) {
		if(0==strcmp(argv[i], "-v")){
			filename = argv[i+1];
		} else if (0 == strcmp(argv[i], "-n")){
			numThreads = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-f")) {
			fps = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-r")) {
			maxOutstandingPerThread = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-b")) {
			bitrateMbps = atof(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-s")) {
			numStreams = atoi(argv[i+1]);
		}
	}

	if (NULL == filename) {
		LOG(ERROR) << "Please provide input video file.";
		printUsage(argv[0]);
		return EXIT_FAILURE;
	}

	if (numThreads > 12) {
		LOG(INFO) << "Max concurrent clients supported = 12. Setting numThreads to 12";
		numThreads = 12;
	}

	if (numStreams > numPhysicalGPUs) {
		LOG(INFO) << "Max concurrent streams supported = " <<numPhysicalGPUs <<". Setting numStreams to " <<numPhysicalGPUs;
		numStreams = numPhysicalGPUs;
	}

	if (fps > 120) {
		std::cout << "Max FPS supported = 120. Setting fps to 120"	<<std::endl;
		fps = 120;
	}

	if (maxOutstandingPerThread > 1000) {
		LOG(INFO) << "Max outstanding requests per thread supported = 1000. Setting to 1000";
		maxOutstandingPerThread = 1000;
	}

	if (bitrateMbps > 6) {
		LOG(INFO) << "Max bitrate supported = 6. Setting to 6";
		bitrateMbps = 6;
	}

	LOG(INFO) << "video file: " << filename;
	LOG(INFO) << "Creating " << numThreads << " threads, each producing frames at "
				<< fps << " FPS.";
	LOG(INFO) << "Each thread can have a maximum of " << maxOutstandingPerThread
				<< " outstanding requests at any time. All other frames will be dropped.";
	LOG(INFO) << "Each thread will encode at " << bitrateMbps << " Mbps.";
	LOG(INFO) << "Press control-c to quit at any point";

	// Create the demuxer (used to read the video stream (h264/h265) from the container (mp4/mkv))
	FFmpegDemuxer demuxer(filename);

	NvPipe_Codec codec;
	uint32_t inWidth = demuxer.GetWidth();
	uint32_t inHeight = demuxer.GetHeight();
	uint32_t bitDepth = demuxer.GetBitDepth();
	AVRational inTimeBase = demuxer.GetTimeBase();

	// Formats supported by NVDEC/CUVID
	// AV_CODEC_ID_MPEG1VIDEO, AV_CODEC_ID_MPEG2VIDEO,
	// AV_CODEC_ID_H264, AV_CODEC_ID_HEVC/AV_CODEC_ID_H265,
	// AV_CODEC_ID_MJPEG, AV_CODEC_ID_MPEG4, AV_CODEC_ID_VC1,
	// AV_CODEC_ID_VP8, AV_CODEC_ID_VP9
	// NvPipe only supports H264 and HEVC, though
	LOG(INFO) << "Timebase numerator/denominator = " <<inTimeBase.num << "/" << inTimeBase.den;

	switch(demuxer.GetVideoCodec())	{
		case AV_CODEC_ID_H264:
			codec = NVPIPE_H264;
			break;
		case AV_CODEC_ID_H265:
			codec = NVPIPE_HEVC;
			break;
		default:
			LOG(ERROR) << "Support for this video codec isn't implemented yet. NVPIPE only supports H264 and H265/HEVC";
			return EXIT_FAILURE;
	}

	// Enable P2P access.
	for (int i = 0; i < numPhysicalGPUs; i++) {
		cudaSetDevice(i);
		for (int j = 0; j < numPhysicalGPUs; j++) {
			if (j == i)
				continue;
			cudaDeviceEnablePeerAccess(j, 0);
		}
	}

	numStreams = std::min(numStreams, numPhysicalGPUs);
	FFmpegStreamer *muxers[numStreams];
	NvPipe* encoders[numStreams];
	NvPipe* decoders[numStreams];
	for (int i = 0; i < numStreams; i++) {
		// Create encoder
		decoders[i] = NvPipe_CreateDecoder(NVPIPE_NV12, codec);
		if (!decoders[i]) {
			LOG(ERROR) << "Failed to create decoder: " << NvPipe_GetError(NULL);
			exit(EXIT_FAILURE);
		}

		encoders[i] = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY, bitrateMbps * 1000 * 1000, fps);
		if (!encoders[i]) {
			LOG(ERROR) << "Failed to create encoder: " << NvPipe_GetError(NULL);
			exit(EXIT_FAILURE);
		}

		std::string outfile = "./scaled" + std::to_string(i) + ".mp4";
		muxers[i] = new FFmpegStreamer(AV_CODEC_ID_H264, inWidth, inHeight, fps, inTimeBase, outfile.c_str());
		if (!muxers[i]) {
			LOG(ERROR) << "Failed to create muxer ";
			exit(EXIT_FAILURE);
		}
	}

	MutexQueue<Frame> compressedFramesQueues[numStreams];
	MutexQueue<Frame> decompressedFramesQueue;

	std::vector<PointerMap<Frame> *> detectedFrameMaps(numStreams);
	std::vector<PointerMap<Frame> *> encodedFrameMaps(numStreams);
	for (int i = 0; i < numStreams; i++){
		detectedFrameMaps[i] = new PointerMap<Frame>;
		encodedFrameMaps[i] = new PointerMap<Frame>;
	}

	uint8_t *compressedFrame = nullptr;
	int compressedFrameSize = 0;
	uint64_t frameNum = 1;
	// Grab compressed frames from the demuxer, and insert them into the FrameMap
	while(demuxer.Demux(&compressedFrame, &compressedFrameSize)) {
		for (int i = 0; i < numStreams; i++) {
			Frame *frame = new Frame;
			frame->frameNum = frameNum++;
			frame->data = new uint8_t[compressedFrameSize];
			std::memcpy(frame->data, compressedFrame, compressedFrameSize);
			frame->frameSize = compressedFrameSize;
			frame->finished = false;
			frame->streamNum = i;
			compressedFramesQueues[i].push_back(*frame);
		}
	}

	// Insert completion frame->
	for(int i=0; i <numStreams; i++) {
		Frame *frame = new Frame;
		frame->frameNum = frameNum++;
		frame->data = nullptr;
		frame->frameSize = -1;
		frame->finished = true;
		frame->streamNum = i;
		compressedFramesQueues[i].push_back(*frame);
	}

	// Launch the detector threads (1 per GPU)
	int numDetectors = numPhysicalGPUs;
	std::vector<std::thread> detectionThreads(numDetectors);
	QueuedDetector detectors[numDetectors];
	MutexQueue<WorkRequest> requestQueue;
	MutexQueue<WorkRequest> completionQueue;
	int cpuMapping[4] = {0,1,12,13};

	// Initialize n detectors where n = numDetectors in the system.
	// Initialization must be done before launching the detection thread.
	// All detectors share one request and completion queue (because detection
	// tends to be the longest part of the workflow)
	for (int i = 0; i < numDetectors; i++) {
		detectors[i].Init(argc, argv, &requestQueue, &completionQueue, i);
		// start a Thread per GPU to run doDetection
		detectionThreads[i] = std::thread(&QueuedDetector::doDetection, &detectors[i]);
	}

	cudaProfilerStart();
	// Launch the pipeline stages in reverse order so the entire pipeline is ready to go
	// (important to for timing measurements)

	std::vector<std::thread> encoderThreads(numStreams);
	for(int i = 0; i < numStreams; i++) {
		encoderThreads[i] = std::thread(&encodeFrame, encoders[i], detectedFrameMaps[i], encodedFrameMaps[i], inWidth, inHeight, i, frameNum-1);
	}

	std::vector<DrawingThread> drawingThreads(numPhysicalGPUs);
	for (int i = 0; i < numPhysicalGPUs; i++){
		drawingThreads[i].Init(i, codec, &completionQueue, detectedFrameMaps, bitrateMbps, fps, inWidth, inHeight);
	}

	std::vector<ResizeThread> resizeThreads(numPhysicalGPUs);
	for (int i = 0; i < numPhysicalGPUs; i++){
		resizeThreads[i].Init(i, codec, &decompressedFramesQueue, &requestQueue, fps, inWidth, inHeight);
	}

	std::vector<std::thread> decodeThreads(numStreams);
	for(int i = 0; i < numStreams; i++) {
		decodeThreads[i] = std::thread(&decodeFrame, decoders[i], &compressedFramesQueues[i],
			&decompressedFramesQueue, inWidth, inHeight, fps, i, frameNum);
	}

	// Try to clean up the FrameMap
	uint64_t outFrameNum = 1;
	while(outFrameNum < frameNum) {
		for(int i = 0; i < numStreams; i++){
			Frame *compressedFrame;
			bool gotFrame = false;
			while(!gotFrame)
				gotFrame = encodedFrameMaps[i]->getElem(&compressedFrame, outFrameNum);
			muxers[i]->Stream((uint8_t *)compressedFrame->data, compressedFrame->frameSize,
								outFrameNum);
			encodedFrameMaps[i]->remove(outFrameNum);
			delete [] compressedFrame->data;
			LOG(INFO) << "Processing frame " << compressedFrame->frameNum << " took "	<< compressedFrame->timer.getElapsedMicroseconds() << " us.";
		}
	}
	cudaProfilerStop();

	LOG(INFO) << "Main thread done. Waiting for other threads to exit";

	for (auto &thread: detectionThreads)
		thread.join();
	for (auto &detector: detectors)
		detector.Shutdown();
	for (auto muxer : muxers)
		delete muxer;
	for (auto map : encodedFrameMaps)
		delete map;
	for (auto map : detectedFrameMaps)
		delete map;

	return 0;
}
