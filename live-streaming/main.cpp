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
#include "nvToolsExt.h"

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

#include "GPUThread.h"

void decodeFrame(NvPipe* decoder, MutexQueue<Frame> *inFrames,
				MutexQueue<Frame> *outFrames, MutexQueue<void *> *gpuFramesQueue,
				int inWidth, int inHeight,int fps, int gpuNum, uint64_t lastFrameNum)
{
	uint64_t frameNum = 0;
	cudaSetDevice(gpuNum);

	while( frameNum < lastFrameNum ) {
		Frame frame;
		while(!inFrames->pop_front(frame));

		frame.timer.reset();
		frame.streamNum = gpuNum;
		frame.decompressedFrameSize = inWidth*inHeight*4;
		frame.deviceNumDecompressed = gpuNum;
		if(!gpuFramesQueue->pop_front(frame.decompressedFrameDevice)){
			LOG(INFO) << "Ran out of buffers. Calling cudaMalloc...";
			cudaMalloc(&frame.decompressedFrameDevice, frame.decompressedFrameSize);
			frame.needsCudaFree = true;
		}

		std::string frameNumString = "Frame " + std::to_string(frameNum);
		frame.nvtxRangeID = nvtxRangeStartA(frameNumString.c_str());
		// Decode the frame
		uint64_t decompressedFrameSize = NvPipe_Decode(decoder,
												(const uint8_t *)frame.data,
												frame.frameSize,
												frame.decompressedFrameDevice,
												inWidth, inHeight);

		if (decompressedFrameSize <= frame.frameSize) {
			std::cerr << "Decode error: " << NvPipe_GetError(decoder)
					<< std::endl;
			exit(-1);
		}

		outFrames->push_back(frame);
		frameNum++;
		usleep(900000.0/fps);
	}
}

void encodeFrame(NvPipe *encoder, PointerMap<Frame> *inFrames,
				PointerMap<Frame> *outFrames, MutexQueue<void *> *gpuFrameBuffers,
				int inWidth, int inHeight, int gpuNum, uint64_t lastFrameNum)
{
	uint64_t frameNum = 0;
	cudaSetDevice(gpuNum);

	while( frameNum < lastFrameNum ) {
		Frame *frame = new Frame;
		bool gotFrame = false;
		while(!gotFrame)
			gotFrame = inFrames->getElem(&frame, frameNum);

		// NvPipe expects us to allocate a buffer for it to output to.. Sigh...
		delete [] frame->data;
		frame->data = new uint8_t[500000];

		// Encode the processed Frame
		uint64_t size = NvPipe_Encode(encoder, frame->decompressedFrameDevice,
							inWidth*4, frame->data, 500000, inWidth, inHeight, false);

		if (0 == size)
			std::cerr << "Encode error: " << NvPipe_GetError(encoder)
					<< std::endl;

		frame->frameSize = size;

		if (frame->needsCudaFree){
			cudaFree(frame->decompressedFrameDevice);
			frame->needsCudaFree = false;
		}
		else {
			cudaMemset(frame->decompressedFrameDevice, 0, frame->decompressedFrameSize);
			gpuFrameBuffers->push_back(frame->decompressedFrameDevice);
			frame->decompressedFrameDevice = nullptr;
		}
		outFrames->insert(frame, frameNum++);
	}
}

void muxThread(int streamID, int lastFrameNum, PointerMap<Frame> *encodedFrameMap,
				FFmpegStreamer *muxer, int fps)
{
	uint64_t outFrameNum = 0;
	uint64_t lastCompletedFrameNum = 0;
	Timer elapsedTime;
	double lastTimerValue = elapsedTime.getElapsedMicroseconds();
	while(outFrameNum < lastFrameNum) {
		Frame *compressedFrame = new Frame;
		bool gotFrame = false;
		while(!gotFrame)
			gotFrame = encodedFrameMap->getElem(&compressedFrame,outFrameNum);
		muxer->Stream((uint8_t *)compressedFrame->data,compressedFrame->frameSize, outFrameNum);
		nvtxRangeEnd(compressedFrame->nvtxRangeID);
		encodedFrameMap->remove(outFrameNum);
		if (outFrameNum%10 == 0){
			LOG(INFO) << "Processing frame " <<compressedFrame->streamNum <<" "
					<< compressedFrame->frameNum << " took "
					<< compressedFrame->timer.getElapsedMicroseconds()
					<< " us.";
		}
		if (outFrameNum%(fps*2) == 0) {
			double elapsedTimeValue =  elapsedTime.getElapsedMicroseconds();
			LOG(INFO) << "Stream " <<streamID <<": Throughput: " << (outFrameNum+1-lastCompletedFrameNum)/((elapsedTimeValue-lastTimerValue)/1000000.0);
			lastTimerValue = elapsedTimeValue;
			lastCompletedFrameNum = outFrameNum;
		}
		outFrameNum++;
	}
}

void printUsage(char *binaryName) {
	LOG(ERROR) << "Usage:" << std::endl
			<< binaryName << " <cfg_file> <weights_file> -v <vid_file> <Opt Args>" <<std::endl
			<< "Optional Arguments:" <<std::endl
			<<	"-s number of video streams (default=1; valid range: 1 to number of GPUs)" <<std::endl
			<<	"-n number of GPUs to use (default=cudaGetDeviceCount; valid range: 1 to cudaGetDeviceCount)" <<std::endl
			<<	"-t number of threads per GPU (default=1; valid range: 1 to 10)" <<std::endl
			<<	"-f fps (default=30fps; valid range: 1 to 120)" <<std::endl
			<<	"-r per_client_max_outstanding_frames (default=100; valid range = 1 to 200)" <<std::endl
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
	int numStreams = 1;
	int numThreadsPerGPU = 1;
	int fps = 30;
	int maxOutstandingFrames = 100;
	float bitrateMbps = 2;

	int numPhysicalGPUs;
	cudaError_t status = cudaGetDeviceCount(&numPhysicalGPUs);
	if (status != cudaSuccess)
		std::cout << "cudaGetDeviceCount Status = " << cudaGetErrorName(status) << std::endl;
	assert(status == cudaSuccess);

	/*status = cudaSetDeviceFlags(cudaDeviceScheduleYield);
	//status = cudaSetDeviceFlags(cudaDeviceScheduleSpin);
	if (status != cudaSuccess)
		std::cout << "cudaGetDeviceCount Status = " << cudaGetErrorName(status) << std::endl;
	assert(status == cudaSuccess);
	*/
	for (int i = 1; i < argc-1; i=i+2) {
		if(0==strcmp(argv[i], "-v")){
			filename = argv[i+1];
		} else if (0 == strcmp(argv[i], "-f")) {
			fps = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-r")) {
			maxOutstandingFrames = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-b")) {
			bitrateMbps = atof(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-s")) {
			numStreams = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-n")) {
			int temp = atoi(argv[i+1]);
			if (temp < numPhysicalGPUs)
				numPhysicalGPUs = temp;
		} else if (0 == strcmp(argv[i], "-t")) {
			numThreadsPerGPU = atoi(argv[i+1]);
		}
	}

	if (NULL == filename) {
		LOG(ERROR) << "Please provide input video file.";
		printUsage(argv[0]);
		return EXIT_FAILURE;
	}

	if (numStreams > numPhysicalGPUs) {
		LOG(INFO) << "Max concurrent streams supported = " <<numPhysicalGPUs
					<<". Setting numStreams to " <<numPhysicalGPUs;
		numStreams = numPhysicalGPUs;
	}

	if (fps > 120) {
		std::cout << "Max FPS supported = 120. Setting fps to 120"
				<<std::endl;
		fps = 120;
	}

	if (numThreadsPerGPU > 10) {
		std::cout << "Max numThreadsPerGPU = 10; setting to 10" << std::endl;
		numThreadsPerGPU = 10;
	}

	if (maxOutstandingFrames > 200) {
		LOG(INFO) << "Max outstanding frames supported = 200. Setting to 200";
		maxOutstandingFrames = 200;
	}

	if (bitrateMbps > 6) {
		LOG(INFO) << "Max bitrate supported = 6. Setting to 6";
		bitrateMbps = 6;
	}

	LOG(INFO) << "video file: " << filename;
	LOG(INFO) << "Creating " << numStreams
				<< " threads, each producing frames at " << fps << " FPS.";
	LOG(INFO) << "The systems supports a maximum of "
				<< maxOutstandingFrames << " outstanding requests at any time. All other frames will be dropped.";
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
	LOG(INFO) << "Timebase numerator/denominator = " <<inTimeBase.num << "/"
				<< inTimeBase.den;

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

	FFmpegStreamer *muxers[numStreams];
	NvPipe* encoders[numStreams];
	NvPipe* decoders[numStreams];
	for (int i = 0; i < numStreams; i++) {
		// Create encoder
		decoders[i] = NvPipe_CreateDecoder(NVPIPE_NV12, codec);
		if (!decoders[i]) {
			LOG(ERROR) << "Failed to create decoder: "
					<< NvPipe_GetError(NULL);
			exit(EXIT_FAILURE);
		}

		encoders[i] = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY,
			bitrateMbps * 1000 * 1000, fps);
		if (!encoders[i]) {
			LOG(ERROR) << "Failed to create encoder: "
					<< NvPipe_GetError(NULL);
			exit(EXIT_FAILURE);
		}

		std::string outfile = "./scaled" + std::to_string(i) + ".mp4";
		muxers[i] = new FFmpegStreamer(AV_CODEC_ID_H264, inWidth, inHeight,
							fps, inTimeBase, outfile.c_str());
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
		encodedFrameMaps[i] = new PointerMap<Frame>;
		detectedFrameMaps[i] = new PointerMap<Frame>;
	}

	// Demux compressed frames, and insert them into the FrameMap
	uint8_t *compressedFrame = nullptr;
	int compressedFrameSize = 0;
	uint64_t frameNum = 0;
	while(demuxer.Demux(&compressedFrame, &compressedFrameSize)) {
		for (int i = 0; i < numStreams; i++) {
			Frame *frame = new Frame;
			frame->frameNum = frameNum;
			frame->data = new uint8_t[compressedFrameSize];
			std::memcpy(frame->data, compressedFrame, compressedFrameSize);
			frame->frameSize = compressedFrameSize;
			frame->streamNum = i;
			compressedFramesQueues[i].push_back(*frame);
		}
		frameNum++;
	}

	int numBuffers = fps*2;
	size_t bufferSize = inWidth*inHeight*4;
	size_t totalBufferSize = numBuffers*bufferSize;
	int numThreads = numThreadsPerGPU * numPhysicalGPUs;
	void *largeBuffers[numThreads];
	MutexQueue<void *> gpuFrameBuffers[numThreads];
	for (int i = 0; i < numPhysicalGPUs; i++) {
		cudaSetDevice(i);
		for (int k = 0; k < numThreadsPerGPU; k++) {
			cudaMalloc(&largeBuffers[i+k], totalBufferSize);
			for (int j = 0; j < numBuffers; j++) {
				void *offset = (void *)((uint8_t *)largeBuffers[i+k]+(bufferSize*j));
				gpuFrameBuffers[i+k].push_back(offset);
			}
		}
	}

	LOG(INFO) << "LAST FRAME = " << frameNum;
	cudaProfilerStart();
	// Launch the pipeline stages in reverse order so the entire pipeline is
	// ready to go (important for timing measurements)

	std::vector<std::thread> encoderThreads(numStreams);
	for(int i = 0; i < numStreams; i++) {
		encoderThreads[i] = std::thread(&encodeFrame, encoders[i],
			detectedFrameMaps[i], encodedFrameMaps[i], &gpuFrameBuffers[i], inWidth, inHeight,
			i, frameNum);
	}

	std::vector<GPUThread> GPUThreads(numThreads);
	int detectorGPUNo[4] = {1,0,3,2};
	// int detectorGPUNo[4] = {0,1,2,3};
	for (int i = 0; i < numPhysicalGPUs; i++) {
		for (int j = 0; j < numThreadsPerGPU; j++) {
			GPUThreads[i*numThreadsPerGPU+j].Init(codec, &decompressedFramesQueue,
						detectedFrameMaps, i*numThreadsPerGPU+j, i, detectorGPUNo[i],
						fps, inWidth, inHeight, numStreams, argc, argv);
		}
	}

	std::vector<std::thread> muxerThreads(numStreams);
	for(int i = 0; i < numStreams; i++) {
		muxerThreads[i] = std::thread(&muxThread, i, frameNum, encodedFrameMaps[i], muxers[i], fps);
	}

	std::vector<std::thread> decoderThreads(numStreams);
	for(int i = 0; i < numStreams; i++) {
		decoderThreads[i] = std::thread(&decodeFrame, decoders[i],
			&compressedFramesQueues[i],	&decompressedFramesQueue, &gpuFrameBuffers[i],
			inWidth, inHeight, fps, i, frameNum);
	}

	LOG(INFO) << "Main thread done. Waiting for other threads to exit";

	for (int i = 0; i < numStreams; i++)
		decoderThreads[i].join();
	LOG(INFO) << "decoderThreads joined!";
	for (int i = 0; i < numStreams; i++)
		encoderThreads[i].join();
	LOG(INFO) << "encodeThreads joined!";
	for (int i = 0; i < numThreads; i++)
		GPUThreads[i].ShutDown();
	for(int i = 0; i < numStreams; i++)
		muxerThreads[i].join();
	LOG(INFO) << "muxerThreads joined!";
	cudaProfilerStop();
	for (auto muxer : muxers)
		delete muxer;
	for (int i = 0; i < numThreads; i++) {
		cudaSetDevice(i);
		cudaFree(largeBuffers[i]);
	}
	for (auto map : encodedFrameMaps)
		delete map;
	for (auto map : detectedFrameMaps)
		delete map;

	return 0;
}
