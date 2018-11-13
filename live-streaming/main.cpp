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
#include <cuda_profiler_api.h>
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

using DarknetWrapper::WorkRequest;
// using DarknetWrapper::DetectionQueue;
using DarknetWrapper::Detector;

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

struct Frame {
	uint8_t *data = nullptr;
	int frameSize = 0;
	uint64_t frameNum;
	Timer timer;
	void *decompressedFrameRGBDevice = nullptr;
	void *decompressedFrameDevice = nullptr;
	bool finished = 0;
}

template <class T>
class FrameMap {
public:

	void insert(T frame, std::uint64_t frameNum)
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		frames.emplace(std::make_pair(frameNum,frame));
		cv.notify_all();
	}

	bool getFrame(T frame, std::uint64_t frameNum)
	{
		std::unique_lock<std::mutex> lock(this->mutex);
		if (frames.empty())
			cv.wait(lock, [this](){ return !this->frames.empty(); });

		auto iterator = frames.find(frameNum);
		if (iterator == frames.end()) {
			return false;
		} else {
			frame = iterator->second;
			return true;
		}
	}

	void remove(std::uint64_t frameNum)
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		frames.erase(frameNum);
	}

	int size() {
		std::lock_guard<std::mutex> lock(this->mutex);
		return frames.size();
	}

private:
	std::unordered_map<std::uint64_t, T> frames;
	std::mutex mutex;
	std::condition_variable cv;

};

class FrameProcessor {
public:
	FrameProcessor(DetectionQueue *requestQueue, DetectionQueue *completionQueue, FrameMap *frames, FFmpegStreamer *muxer, float bitrateMbps, int targetFPS, int inWidth, int inHeight, int maxOutstandingPerThread, int threadID)
	{
		this->requestQueue = requestQueue;
		this->completionQueue = completionQueue;
		this->frames = frames;
		this->muxer = muxer;
		this->bitrateMbps = bitrateMbps;
		this->targetFPS = targetFPS;
		this->inWidth = inWidth;
		this->inHeight = inHeight;
		this->maxOutstandingPerThread = maxOutstandingPerThread;
		this->threadID = threadID;
	}

	void Init(int gpuNum, NvPipe_Codec codec)
	{
		cudaSetDevice(gpuNum);

		// Create decoder
		decoder = NvPipe_CreateDecoder(NVPIPE_NV12, codec);
		if (!decoder) {
			LOG(ERROR) << "Failed to create decoder: " << NvPipe_GetError(NULL) << std::endl;
			exit(EXIT_FAILURE);
		}

		// Create encoder
		encoder = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY, bitrateMbps * 1000 * 1000, targetFPS);
		if (!encoder) {
			LOG(ERROR) << "Failed to create encoder: " << NvPipe_GetError(NULL) << std::endl;
			exit(EXIT_FAILURE);
		}

	}

	void DecodeAndResize()
	{
		void *decompressedFrameDevice = nullptr;
		void *decompressedFrameRGBDevice = nullptr;
		void *scaledFrameNoPad = nullptr;
		void *scaledFramePadded = nullptr;
		void *scaledPaddedPlanarS = nullptr;

		cudaMalloc(&decompressedFrameDevice, inWidth*inHeight*4);
		cudaMalloc(&decompressedFrameRGBDevice, inWidth*inHeight*sizeof(float3));
		cudaMalloc(&scaledFrameNoPad, noPadWidth*noPadHeight*sizeof(float3));
		cudaMalloc(&scaledFramePadded, netWidth*netHeight*sizeof(float3));
		cudaMalloc(&scaledPaddedPlanarS, netWidth*netHeight*sizeof(float)*3);

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

		scaledPaddedPlanar[0] = (float *)scaledPaddedPlanarS;
		scaledPaddedPlanar[1] = (float *)(scaledPaddedPlanarS+netWidth*netHeight*sizeof(float));
		scaledPaddedPlanar[2] = (float *)(scaledPaddedPlanarS+netWidth*netHeight*sizeof(float));

		while(true) {
			Frame frame;
			bool gotImage = false;
			while (!gotImage) {
				gotImage = frames->getFrame(frame, this->currentFrame);
			}

			// Check if this is a frame that was inserted to indicate that
			// there is no more work.
			if (frame.finished == true){
				// Add a completion WorkRequest to the completion queue to signal to the
				// completion thread to finish.
				WorkRequest work;
				work.done = false;
				work.finished = true;
				work.dets = nullptr;
				work.nboxes = 0;
				work.classes = 0;
				work.tag = nullptr;
				requestQueue.push_back(work);
				break;
			}

			if (outstandingRequests.load(std::memory_order_acquire) > maxOutstandingPerThread) {
				numDropped++;
			} else {
				// Allocate GPU memory and copy the compressed Frame to it
				void *compressedFrameDevice = nullptr;
				cudaMalloc(&compressedFrameDevice, frame.frameSize);
				cudaMemcpy(compressedFrameDevice, frame.data, frame.frameSize, cudaMemcpyHostToDevice);

				cudaMalloc(&frame.decompressedFrameDevice, inWidth*inHeight*4);
				cudaMalloc(&frame.decompressedFrameRGBDevice, inWidth*inHeight*sizeof(float3));

				// Decode the frame
				uint64_t decompressedFrameSize = NvPipe_Decode(decoder, compressedFrame, compressedFrameSize, frame.decompressedFrameDevice, inWidth, inHeight);
				if (decompressedFrameSize <= compressedFrameSize) {
					std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;
					exit(-1);
				}

				cudaError_t status;
				NppStatus nppStatus;

				// Convert to RGB from NV12
				status = cudaNV12ToRGBf(static_cast<uint8_t *>(frame.decompressedFrameDevice),
													static_cast<float3 *>(frame.decompressedFrameRGBDevice),
													(size_t)inWidth,
													(size_t)inHeight);
				if (status != cudaSuccess)
					std::cout << "cudaNV12ToRGBf Status = " << cudaGetErrorName(status) << std::endl;
				assert(status == cudaSuccess);

				// Scale the frame to noPadWidth;noPadHeight
				status = cudaResizeRGB(static_cast<float3 *>(frame.decompressedFrameRGBDevice),
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
				work.tag = &frame;

				// Put packet in processing queue
				requestQueue->push_back(work);
				// Increment outstandingRequests
				outstandingRequests.fetch_add(1, std::memory_order_release);

				cudaFree(compressedFrameDevice);
			}

			this->currentFrame++;
			// Account for the time spent processing the packet...
			usleep((1000000/fps) - timer.getElapsedMicroseconds);
		} // while(true)

		cudaFree(compressedFrameDevice);
		cudaFree(scaledFrameNoPad);
		cudaFree(scaledFramePadded);
		cudaFree(scaledPaddedPlanarS);
	}

	void DrawBoxesAndEncode()
	{
		const float4 overlayColor = {75.0, 156.0, 211.0,120.0};
		// NvPipe expects us to allocate a buffer for it to output to.. Sigh...
		uint8_t *compressedOutFrame = new uint8_t[200000];

		while(true) {
			WorkRequest work;
			completionQueue->pop_front(work);
			// Break out of the loop if this is the completion signal.
			if(work.done == false && work.finished == true)
				break;
			assert(work.done == true);
			assert(work.dets != nullptr);

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

			if (numObjects >0) {

				void *boundingBoxesDevice = nullptr;
				cudaMalloc(&boundingBoxesDevice, numObjects*sizeof(float4));
				cudaMemcpy(boundingBoxesDevice, boundingBoxes.data(), numObjects*sizeof(float4), cudaMemcpyHostToDevice);

				status = cudaRectOutlineOverlay((float3 *)frame.decompressedFrameRGBDevice,
												(float3 *)frame.decompressedFrameRGBDevice,
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

			// Convert to uchar4 BGRA8 that the Encoder wants
			status = cudaRGBToBGRA8((float3 *)frame.decompressedFrameRGBDevice,
									(uchar4*) frame.decompressedFrameDevice,
									inWidth,
									inHeight);

			if (status != cudaSuccess)
				std::cout << "cudaRGBToRGBA8 Status = " << cudaGetErrorName(status)	<< std::endl;
			assert(status == cudaSuccess);

			// Encode the processed Frame
			uint64_t size = NvPipe_Encode(encoder, frame.decompressedFrameDevice, inWidth*4, compressedOutFrame, 200000, inWidth, inHeight, false);
			if (0 == size)
				std::cerr << "Encode error: " << NvPipe_GetError(encoder) << std::endl;

			// MUX the frame
			muxer->Stream(compressedOutFrame, size, frame.frameNum);
			cudaFree(decompressedFrameRGBDevice);
			cudaFree(decompressedFrameDevice);
			this->lastCompletedFrame = frame.frameNum;
			this->decrementOutstanding();
			LOG(INFO) << "Processing frame " << frame.frameNum << " took "
						<< timer.getElapsedMicroseconds() << " us.";
		}
		delete[] compressedOutFrame;
	}

	void decrementOutstanding()
	{
		outstandingRequests.fetch_sub(1, std::memory_order_release);
	}

	std::uint64_t getLastCompleted()
	{
		return lastCompletedFrame;
	}

	std::uint64_t getNumDropped()
	{
		return numDropped;
	}

	~FrameProcessor()
	{
		NvPipe_Destroy(encoder);
		NvPipe_Destroy(decoder);
	}

private:
	int netWidth = 416;
	int netHeight = 416;

	int maxOutstandingPerThread;
	int threadID;

	// Variables we operate on
	float bitrateMbps;
	int targetFPS;
	int inWidth;
	int inHeight;
	std::uint64_t currentFrame;
	std::uint64_t lastCompletedFrame;
	std::uint64_t numDropped;
	std::atomic<unsigned int> outstandingRequests;

	// Objects (or pointers to)
	// Create one output stream writer wrapper per thread
	FFmpegStreamer *muxer;
	std::thread thread;
	NvPipe* decoder;
	NvPipe* encoder;
	FrameMap *frames;
	DetectionQueue *requestQueue;
	DetectionQueue *completionQueue;
};

void printUsage(char *binaryName) {
	LOG(ERROR) << "Usage:" << std::endl
			<< binaryName << " <cfg_file> <weights_file> -v <vid_file>
			Optional Arguments:
				-n number-of-clients(default=1; valid range: 1 to 12)
				-f fps (default=30fps; valid range: 1 to 120)
				-r per_client_max_outstanding_requests (default=90; valid range = 1 to 1000)
				-b bit rate of output video (in Mbps; default=2; valid range = 1 to 6;)";
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
	int fps = 30;
	int maxOutstandingPerThread = 90;
	float bitrateMbps = 2;

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

	LOG(INFO) << "video file:" << filename;
	LOG(INFO) << "Creating " << numThreads << "threads, each producing frames at "
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
	// In the cross-GPU case, the nth gpu runs decode/encode and the n+1th gpu runs detection...
	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1, 0);
	cudaDeviceEnablePeerAccess(2, 0);
	cudaDeviceEnablePeerAccess(3, 0);
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0, 0);
	cudaDeviceEnablePeerAccess(2, 0);
	cudaDeviceEnablePeerAccess(3, 0);
	cudaSetDevice(2);
	cudaDeviceEnablePeerAccess(0, 0);
	cudaDeviceEnablePeerAccess(1, 0);
	cudaDeviceEnablePeerAccess(3, 0);
	cudaSetDevice(3);
	cudaDeviceEnablePeerAccess(0, 0);
	cudaDeviceEnablePeerAccess(1, 0);
	cudaDeviceEnablePeerAccess(2, 0);

	std::std::vector<FFmpegStreamer> muxers;
	for( int i=0; i<numThreads; ++i ){
		char filename[20];
		strcpy(filename,"./scaled");
		strcat(filename, itoa(i));
		strcat(filename, ".mp4");
		muxers.push(muxer(AV_CODEC_ID_H264, inWidth, inHeight, fps, inTimeBase, filename));
	}

	FrameMap<Frame> frames;

	// Launch the detector threads (1 per GPU)
	std::vector<std::thread> detectionThreads(4);
	QueuedDetector detectors[4];
	DetectionQueue requestQueues[4];
	DetectionQueue completionQueues[4];
	int cpuMapping[4] = {0,1,12,13};

	// Initialize n detectors where n = numGPUs in the system.
	// We should automatically figure out numGPUs, but meh...
	// Initialization must be done before launching the detection thread.
	for (int i = 0; i < 4; i++) {
		detectors[i].Init(argc, argv, &requestQueues[i], &completionQueues[i], i);
		// start a Thread per GPU to run doDetection
		detectionThreads[i] = std::thread(&QueuedDetector::doDetection, &detector[i]);
		// Create a cpu_set_t object representing a set of CPUs. Clear it and mark
		// only CPU i as set.
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		// 0 -> 0; 1 -> 24; 2 -> 12; 3 -> 36;
		CPU_SET(cpuMapping[i], &cpuset);
		int rc = pthread_setaffinity_np(detectionThreads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
		if (rc != 0)
			std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
	}

	std::vector<FrameProcessor> frameProcessors;
	std::vector<std::thread> decoders;
	std::vector<std::thread> encoders;
	for (int i = 0; i < numThreads; i++){
		int j = i%4;
		frameProcessors.push(FrameProcessor(requestQueues[j], completionQueues[j], frames,
			muxer[i], bitrateMbps, targetFPS, inWidth, inHeight, maxOutstandingPerThread, i));
		// Use the next GPU for decode and encode
		frameProcessor[i].Init((j+1)%4, codec);
		decoders[i] = std::thread(&FrameProcessor::DecodeAndResize, &frameProcessor[i]);
		encoders[i] = std::thread(&FrameProcessor::DrawBoxesAndEncode, &frameProcessor[i]);
	}

	uint8_t *compressedFrame = nullptr;
	int compressedFrameSize = 0;

	uint64_t frameNum = 1;
	cudaProfilerStart();
	// Grab compressed frames from the demuxer, and insert them into the FrameMap
	while(true) {
		timer.reset();
		Frame frame;
		frame.timer.reset();
		frame.frameNum = frameNum;

		if(!demuxer.Demux(&compressedFrame, &compressedFrameSize)) {
			frame.data = nullptr;
			frame.frameSize = -1;
			frame.finished = true;
			frames.insert(image, frameNum);
			break;
		}

		frame.data = compressedFrame;
		frame.frameSize = compressedFrameSize;
		frame.finished = false;
		frames.insert(image, frameNum++);
	}

	// Try to clean up the FrameMap
	while(true) {
		std::uint64_t minProcessedFrameNum = requestThreads[0].getCurrentFrame()-1;
		for (int i = 1; i < numThreads; i++) {
			minProcessedFrameNum = std::min(requestThreads[i].getCurrentFrame()-1, minProcessedFrameNum);
		}
		auto removeUpto = std::max(minProcessedFrameNum, lastFrameDequeued);
		while (lastFrameDequeued < removeUpto) {
			lastFrameDequeued++;
			frames.remove(lastFrameDequeued);
		}

		std::cout << "LastFrameDequeued = " << lastFrameDequeued <<std::endl;
		for (int i = 0; i < numThreads; i++) {
			std::cout << "Thread " << i << " dropped " << requestThreads[i].getNumDropped() << " frames so far." <<std::endl;
		}

		// Break out of this loop if the last frame (the fake one) was processed.
		if (lastFrameDequeued == frameNum)
			break;

		// Rate-limit so that we don't consume too much CPU...
		sleep(2);
	}
	cudaProfilerStop();

	for (auto &thread: decoders)
		thread.join();
	for (auto &thread: detectionThreads)
		thread.join();
	for (auto &thread: encoders)
		thread.join();
	for (auto &detector: detectors)
		detector.Shutdown();

	return 0;
}
