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
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <functional>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>

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

// Gotta create the logger before including FFmpegStreamer/Demuxer
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

#include <openssl/conf.h>
#include <openssl/evp.h>
#include <openssl/err.h>

#define KEY_SIZE = 256;
#define BLOCK_SIZE = AES_BLOCK_SIZE;
#define IV_SIZE = 128;
#define MAX_FILE_SIZE = 100000;

// The program is composed as follows:
// main loads a video into memory and then demuxes (takes frames out of the
// container, e.g., MP4 --- MP4 is not a binary encoding but a data
// representation format, i.e., it contains a bunch of H.264 encoded pictures
// and some metadata about how to access them, their order, etc.) them into a
// mutex-protected queue. All the frames are demuxed into the queue before we
// do any other processing to ensure we're not waiting on disk I/O. This of
// course would be different in a server that is receiving videos over the
// network. The demuxed (but still encoded) image is put into a Frame object;
//
// These Frames is then passed to the decodeFrame thread(s), this thread uses
// Nvidia's H.264 decoder to decode the frame into raw RGBA data.
//
// The raw data is put into a buffer in the Frame object and passed to the
// poorly named 'GPUthread'. This thread does a bunch of transformations on the
// image and then passes it through the object detector and then draws a
// bounding box for the detected objects onto the raw frame;
//
// After all that has been done the encodeFrame thread(s) take the processed
// frames, and pass them through the Nvidia HW encoder on the GPU and then the
// MuxerThread takes these encoded frames and muxes them into an mp4 container;
//
// Note: The program may be processing multiple copies of the same video at
// once (to create more work); these are referred to as streams;
// One encoder, encoderThread, decoder, decoderThread, muxer, and muxerThread
// are created per Stream;

/*This function is used to launch a thread that decodes a frame using Nvidia's
* on-GPU decoders (using the NVPipe API).
* Inputs:
* NvPipe* decoder --- the NVPipe decoder object (initialized in main)
* MutexQueue<Frame> *inFrames --- A mutex-protected input queue of Frames
* MutexQueue<Frame> *outFrames --- Similar, but for processed Frames
* Look under utils/ for the MutexQueue and Frame data types;
* MutexQueue<void *> *gpuFramesQueue --- This is a queue of pre-allocated
*   buffers on the GPU, these buffers are what NVPipe writes the decoded image
*   into on the GPU.
* int inWidth --- width of the input image
* int inHeight --- height of the input image
* int fps --- the target FPS we're trying to run our pipeline at (used to
*   determine how long this thread should sleep)
* int gpuNum --- The GPU to decode the frame on. This system was built to do
*   the decoding and encoding on 1 GPU and the Object Detection on another
*   GPU, but is configurable so all of these operations can run on one GPU.
* uint64_t lastFrameNum --- The total number of frames the system is going to
*   process, so that we know when to terminate; the code currently reads a
*   video from a file, this is predetermined; this will need to change if
*   we're running a long running service that accepts videos from the internet.
*/
void decodeFrame(NvPipe* decoder, MutexQueue<Frame> *inFrames,
  MutexQueue<Frame> *outFrames, MutexQueue<void *> *gpuFramesQueue,
  int inWidth, int inHeight,int fps, int gpuNum, uint64_t lastFrameNum)
{
  uint64_t frameNum = 0;
  // Used to set the GPU that this thread will be running its operations on.
  cudaSetDevice(gpuNum);

  while( frameNum < lastFrameNum ) {
    Frame frame;
    // Busy loop until we get a frame to process;
    while(!inFrames->pop_front(frame));

    // Some book keeping for the frame
    frame.timer.reset();
    frame.streamNum = gpuNum;
    frame.decompressedFrameSize = inWidth*inHeight*4;
    frame.deviceNumDecompressed = gpuNum;
    // Grab a pre-allocated buffer to use;
    // returned to the queue by the encode thread;
    if(!gpuFramesQueue->pop_front(frame.decompressedFrameDevice)){
      LOG(INFO) << "Ran out of buffers. Calling cudaMalloc...";
      cudaMalloc(&frame.decompressedFrameDevice, frame.decompressedFrameSize);
      frame.needsCudaFree = true;
    }

    // Give frame an ID so we can visualize the GPU processing using NVTools
    std::string frameNumString = "Frame " + std::to_string(frameNum);
    frame.nvtxRangeID = nvtxRangeStartA(frameNumString.c_str());

    // Decode the frame on the GPU using NVPipe API;
    uint64_t decompressedFrameSize = NvPipe_Decode(decoder,
                                            (const uint8_t *)frame.data,
                                            frame.frameSize,
                                            frame.decompressedFrameDevice,
                                            inWidth, inHeight);

    if (decompressedFrameSize <= frame.frameSize) {
        std::cerr << "Decode error: " << NvPipe_GetError(decoder) << std::endl;
        exit(-1);
    }

    outFrames->push_back(frame);
    frameNum++;
    usleep(900000.0/fps);
  }
}


/*
* This function is used to launch a thread that encodes a processed frame
* using Nvidia's on-GPU encoders (using the NVPipe API).
* Inputs:
*   NvPipe* encoder --- the NVPipe encoder object (initialized in main)
*   MutexQueue<Frame> *inFrames --- A mutex-protected input queue of Frames
*   MutexQueue<Frame> *outFrames --- Similar, but for processed Frames
*   Look under utils/ for the MutexQueue and Frame data types;
*   MutexQueue<void *> *gpuFramesQueue --- This is a queue of pre-allocated
*     buffers on the GPU, these buffers are what NVPipe grabs the decoded image
*     from on the GPU. This function releases these buffers onto this queue;
*   int inWidth --- width of the input image
*   int inHeight --- height of the input image
*   int gpuNum --- The GPU to decode the frame on. This system was built to do
*     the decoding and encoding on 1 GPU and the Object Detection on another
*     GPU, but is configurable so all of these operations can run on one GPU.
*   uint64_t lastFrameNum --- The total number of frames the system is going to
*     process, so that we know when to terminate; the code currently reads a
*     video from a file, this is predetermined; this will need to change if
*     we're running a long running service that accepts videos from the
*     internet.
*/
void encodeFrame(NvPipe *encoder, PointerMap<Frame> *inFrames,
  PointerMap<Frame> *outFrames,MutexQueue<void *> *gpuFrameBuffers,
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
            gpuFrameBuffers->push_back(frame->decompressedFrameDevice);
            frame->decompressedFrameDevice = nullptr;
        }
        outFrames->insert(frame, frameNum++);
    }
}

//takes dara from decryptedFile, ecrypts it, and writes it into "filename"
//input: name of file to write encrypted data into
//return: -1 if error, 1 otherwise
int encryptFileCBC(char*& filename)
{
  FILE* keyFile;
  FILE* ivFile;
  FILE* fileToEncrypt;
  AES_KEY key;
  unsigned char* keyString = (unsigned char*)(malloc(KEY_SIZE*sizeof(char)));
  unsigned char* iv = (unsigned char*)(malloc(BLOCK_SIZE*sizeof(char)));
  fileToEncrypt = fopen("decryptedFile", "r");
  keyFile = fopen("aesKey", "r");
  ivFile = fopen("aesIv", "r");
  fread(keyString, KEY_SIZE/8, 1, keyFile);
  fread(iv, BLOCK_SIZE, 1, keyFile);
  AES_set_encrypt_key((const unsigned char*)keyString, KEY_SIZE, &key);

  fseek(fileToEncrypt, 0, SEEK_END);
  long fsize = ftell(fileToEncrypt);
  if(fsize == 0){
    fclose(fileToEncrypt);
    fclose(ivFile);
    fclose(keyFile);

    free(keyString);
    free(iv);
    free(cipherText);
    free(plainText);
    return -1;
  }
  fseek(fileToEncrypt, 0, SEEK_SET);
  unsigned char* plainText = (unsigned char*)(malloc((fsize+1)*sizeof(char)));
  fread(plainText, sizeof(char), fsize, fileToEncrypt);
  unsigned char* cipherText = (unsigned char*)(malloc((fsize+BLOCK_SIZE+1)*sizeof(char)));
  //encrypt plaintext and move data into ciphertext
  AES_cbc_encrypt(plainText, cipherText, fsize, &key, iv, AES_ENCRYPT);
  //write to"decryptedFile"
  FILE* encryptedFile = fopen(filename, "w+"); //file descript

  fwrite(cipherText, sizeof(char), strlen((char*)cipherText), encryptedFile);
  fclose(encryptedFile);
  fclose(fileToEncrypt);
  fclose(ivFile);
  fclose(keyFile);

  free(keyString);
  free(iv);
  free(cipherText);
  free(plainText);
  return 1;
}

//takes a file of given name, decrypts data and writes it into "decryptedFile"
//input: name of file containing encrypted data
//return: -1 if error, 1 otherwise
int decryptFileCBC(char*& filename)
{
  FILE* keyFile;
  FILE* ivFile;
  FILE* fileToDecrypt;
  AES_KEY key;
  unsigned char* keyString = (unsigned char*)(malloc(KEY_SIZE*sizeof(char)));
  unsigned char* iv = (unsigned char*)(malloc(BLOCK_SIZE*sizeof(char)));
  fileToDecrypt = fopen(filename, "w+");
  keyFile = fopen("aesKey", "r");
  ivFile = fopen("aesIv", "r");
  fread(keyString, KEY_SIZE/8, 1, keyFile);
  fread(iv, BLOCK_SIZE, 1, keyFile);
  AES_set_decrypt_key((const unsigned char*) keyString, KEY_SIZE, &key);

  fseek(fileToDecrypt, 0, SEEK_END);
  long fsize = ftell(fileToDecrypt);
  if(fsize == 0){
    fclose(fileToEncrypt);
    fclose(ivFile);
    fclose(keyFile);

    free(keyString);
    free(iv);
    free(cipherText);
    free(plainText);
    return -1;
  }
  fseek(fileToDecrypt, 0, SEEK_SET);
  unsigned char* cipherText = (unsigned char*)(malloc((fsize+1)*sizeof(char)));
  fread(cipherText, sizeof(char), fsize, fileToDecrypt);

  unsigned char* plainText = (unsigned char*)(malloc((fsize+BLOCK_SIZE+1)*sizeof(char)));
  //decrypt cyphertext and move data into plaintext
  AES_cbc_encrypt(cipherText, plainText, fsize, &key, iv, AES_DECRYPT);
  //write to"decryptedFile"
  FILE* decryptedFile = fopen("decryptedFile", "w+");
  fwrite(plainText, sizeof(char), strlen((char*)plainText), decryptedFile);
  fclose(decryptedFile);
  fclose(fileToDecrypt);
  fclose(ivFile);
  fclose(keyFile);

  free(keyString);
  free(iv);
  free(cipherText);
  free(plainText);
  return 1;
}

/*
* This function is used to mux an encoded frame into an MP4 container
* Inputs:
*   int streamID --- StreamID identifies the frames this thread should mux
*     into a single video so we don't end up mixing up videos;
*     Note: we're duplicating the same video multiple times into 'streams'
*   int lastFrameNum --- The total number of frames (per video stream);
      A hack that is used to determine when to terminate; All video streams are the same video file; so there is only one lastFrameNum;
*   PointerMap<Frame> *encodedFrameMap --- where to pop the encodedFrames
*     from; One per stream;
*   FFmpegStreamer *muxer --- Muxer Object
*   int fps --- target framerate;
*/
void muxThread(int streamID, int lastFrameNum,
  PointerMap<Frame> *encodedFrameMap, FFmpegStreamer *muxer, int fps)
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
  LOG(ERROR) << "Usage:" << std::endl;
    << binaryName << " <cfg_file> <weights_file> -v <vid_file> <Opt Args>" <<std::endl
    << "Optional Arguments:" <<std::endl
    <<  "-s number of video streams (default=1; valid range: 1 to number of GPUs)" <<std::endl
    <<  "-n number of GPUs to use (default=cudaGetDeviceCount; valid range: 1 to cudaGetDeviceCount)" <<std::endl
    <<  "-f fps (default=30fps; valid range: 1 to 120)" <<std::endl
    <<  "-r per_client_max_outstanding_frames (default=100; valid range = 1 to 200)" <<std::endl
    <<  "-m mem_to_burn_in_bytes" <<std::endl
    <<  "-b bit rate of output video (in Mbps; default=2; valid range = 1 to 6;)" <<std::endl;
}

int getNumPhysicalGPUs() {
  int numPhysicalGPUs = 0;
  cudaError_t status = cudaGetDeviceCount(&numPhysicalGPUs);
  if (status != cudaSuccess)
    std::cout << "cudaGetDeviceCount Status = "
              << cudaGetErrorName(status) << std::endl;
    assert(status == cudaSuccess);
  return numPhysicalGPUs;
}

void parseCommandLineArgs(int argc, char* argv[], char*& filename, char*& originalFilename
  int& numStreams, int& fps, int& maxOutstandingFrames, float& bitrateMbps,
  size_t& memToBurn, int& numPhysicalGPUs)
{
  filename = "decryptedFile";
  for (int i = 1; i < argc-1; i=i+2) {
    if(0==strcmp(argv[i], "-v")){
      originalFilename = argv[i+1];
    } else if (0 == strcmp(argv[i], "-f")) {
      fps = atoi(argv[i+1]);
    } else if (0 == strcmp(argv[i], "-r")) {
      maxOutstandingFrames = atoi(argv[i+1]);
    } else if (0 == strcmp(argv[i], "-m")) {
      memToBurn = atoi(argv[i+1]);
    } else if (0 == strcmp(argv[i], "-b")) {
      bitrateMbps = atof(argv[i+1]);
    } else if (0 == strcmp(argv[i], "-s")) {
      numStreams = atoi(argv[i+1]);
    } else if (0 == strcmp(argv[i], "-n")) {
      int temp = atoi(argv[i+1]);
      numPhysicalGPUs = (temp < numPhysicalGPUs) ? temp : numPhysicalGPUs;
    }
  }

  if (NULL == originalFilename) {
      LOG(ERROR) << "Please provide input video file.";
      printUsage(argv[0]);
      return EXIT_FAILURE;
  }

  if(decryptFile(originalFilename) == -1){
      LOG(ERROR) << "Error decrypting file.";
      return EXIT_FAILURE;
  }

  if (numStreams > numPhysicalGPUs) {
    LOG(INFO) << "Max concurrent streams supported = " <<numPhysicalGPUs
      <<". Setting numStreams to " <<numPhysicalGPUs <<std::endl;;
    numStreams = numPhysicalGPUs;
  }

  if (fps > 120) {
    std::cout << "Max FPS supported = 120. Setting fps to 120" <<std::endl;
    fps = 120;
  }

  if (maxOutstandingFrames > 200) {
    LOG(INFO) << "Max outstanding frames supported = 200. Setting to 200"
      <<std::endl;
    maxOutstandingFrames = 200;
  }

  if (bitrateMbps > 6) {
    LOG(INFO) << "Max bitrate supported = 6. Setting to 6" <<std::endl;;
    bitrateMbps = 6;
  }
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
  char *originalFilename;
  int numStreams = 1;
  int fps = 30;
  int maxOutstandingFrames = 100;
  float bitrateMbps = 2;
  size_t memToBurn = 0*1024*1024;
  int numPhysicalGPUs = getNumPhysicalGPUs();

  parseCommandLineArgs(filename, originalFilename, numStreams, fps, maxOutstandingFrames,
    bitrateMbps, memToBurn, numPhysicalGPUs)

  LOG(INFO) << "video file: " << filename;
  LOG(INFO) << "Creating " << numStreams
              << " threads, each producing frames at " << fps << " FPS.";
  LOG(INFO) << "The systems supports a maximum of "
              << maxOutstandingFrames << " outstanding requests at any time. All other frames will be dropped.";
  LOG(INFO) << "Each thread will encode at " << bitrateMbps << " Mbps.";
  LOG(INFO) << "Press control-c to quit at any point";

  // Create the demuxer
  // (used to read the video stream (h264/h265) from the container (mp4/mkv))
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
  //
  // Important: NvPipe only supports H264 and HEVC, though
  LOG(INFO) << "Timebase numerator/denominator = " <<inTimeBase.num << "/"
              << inTimeBase.den;

  switch(demuxer.GetVideoCodec()) {
    case AV_CODEC_ID_H264:
      codec = NVPIPE_H264;
      break;
    case AV_CODEC_ID_H265:
      codec = NVPIPE_HEVC;
      break;
    default:
      LOG(ERROR) << "Support for this video codec isn't implemented yet.  NVPIPE only supports H264 and H265/HEVC";
      return EXIT_FAILURE;
  }

  // Enable Peer2Peer access among GPUs, if >1 GPU.
  for (int i = 0; i < numPhysicalGPUs; i++) {
    cudaSetDevice(i);
    for (int j = 0; j < numPhysicalGPUs; j++) {
      if (j == i) continue;
      cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  // This exists as a way to induce some memory pressure on the GPU;
  // Usually, not used;
  if (memToBurn != 0) {
    void * reservedMem;
    cudaMallocHost(&reservedMem, memToBurn);
  }

  // Create the per-video-stream objects;
  FFmpegStreamer *muxers[numStreams];
  NvPipe* encoders[numStreams];
  NvPipe* decoders[numStreams];
  // These Queues are used to store the perStream compressed frames to be muxed
  MutexQueue<Frame> compressedFramesQueues[numStreams];
  for (int i = 0; i < numStreams; i++) {
    decoders[i] = NvPipe_CreateDecoder(NVPIPE_NV12, codec);
    if (!decoders[i]) {
      LOG(ERROR) << "Failed to create decoder: " << NvPipe_GetError(NULL);
      exit(EXIT_FAILURE);
    }

    encoders[i] = NvPipe_CreateEncoder(NVPIPE_RGBA32, codec, NVPIPE_LOSSY,
                                        bitrateMbps * 1000 * 1000, fps);
    if (!encoders[i]) {
      LOG(ERROR) << "Failed to create encoder: " << NvPipe_GetError(NULL);
      exit(EXIT_FAILURE);
    }

    std::string outfile = "./scaled" + std::to_string(i) + ".mp4";
    muxers[i] = new FFmpegStreamer(AV_CODEC_ID_H264, inWidth, inHeight,
                                    fps, inTimeBase, outfile.c_str());
    if (!muxers[i]) {
      LOG(ERROR) << "Failed to create muxer.";
      exit(EXIT_FAILURE);
    }
  }

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

  int numBuffers = fps*4;
  size_t bufferSize = inWidth*inHeight*4;
  size_t totalBufferSize = numBuffers*bufferSize;
  void *largeBuffers[numStreams];
  MutexQueue<void *> gpuFrameBuffers[numStreams];
  for (int i = 0; i < numStreams; i++) {
    cudaSetDevice(i);
    cudaMalloc(&largeBuffers[i], totalBufferSize);
    for (int j = 0; j < numBuffers; j++) {
      void *offset = (void *)((uint8_t *)largeBuffers[i]+(bufferSize*j));
      gpuFrameBuffers[i].push_back(offset);
    }
  }

  LOG(INFO) << "LAST FRAME = " << frameNum;
  cudaProfilerStart();

  // Launch the pipeline stages in reverse order so the entire pipeline is
  // ready to go (important for timing measurements)

  std::vector<std::thread> encoderThreads(numStreams);
  for(int i = 0; i < numStreams; i++) {
    encoderThreads[i] = std::thread(&encodeFrame, encoders[i],
      detectedFrameMaps[i], encodedFrameMaps[i], &gpuFrameBuffers[i], inWidth,
      inHeight, i, frameNum);
  }

  std::vector<GPUThread> GPUThreads(numPhysicalGPUs);
  int detectorGPUNo[4] = {1,0,3,2};
  // int detectorGPUNo[4] = {0,1,2,3};
  for (int i = 0; i < numPhysicalGPUs; i++) {
      GPUThreads[i].Init(codec, &decompressedFramesQueue,
                      detectedFrameMaps, i, detectorGPUNo[i],
                      fps, inWidth, inHeight, numStreams, argc, argv);
  }

  std::vector<std::thread> muxerThreads(numStreams);
  for(int i = 0; i < numStreams; i++) {
      muxerThreads[i] = std::thread(&muxThread, i, frameNum, encodedFrameMaps[i], muxers[i], fps);
  }

#ifdef PROFILE
  // Launch profiler
  std::stringstream s;
  s << getpid();
  pid_t pid = fork();
  if (pid == 0) {
      exit(execl("/usr/bin/perf","perf","record","-o","perf.data","-p",s.str().c_str(),nullptr));
  }
#endif

  std::vector<std::thread> decoderThreads(numStreams);
  for(int i = 0; i < numStreams; i++) {
      decoderThreads[i] = std::thread(&decodeFrame, decoders[i],
          &compressedFramesQueues[i], &decompressedFramesQueue, &gpuFrameBuffers[i],
          inWidth, inHeight, fps, i, frameNum, numPhysicalGPUs);
  }

  // Clean up spawned threads;
  LOG(INFO) << "Main thread done. Waiting for other threads to exit";
  for (int i = 0; i < numStreams; i++)
      decoderThreads[i].join();
  LOG(INFO) << "decoderThreads joined!";
  for (int i = 0; i < numStreams; i++)
      encoderThreads[i].join();
  LOG(INFO) << "encodeThreads joined!";
  for (int i = 0; i < numPhysicalGPUs; i++)
      GPUThreads[i].ShutDown();
  for(int i = 0; i < numStreams; i++)
      muxerThreads[i].join();
  LOG(INFO) << "muxerThreads joined!";
  cudaProfilerStop();

  // Clean up allocated objects;
  for (auto muxer : muxers)
      delete muxer;
  for (int i = 0; i < numPhysicalGPUs; i++) {
      cudaSetDevice(i);
      cudaFree(largeBuffers[i]);
  }
  for (auto map : encodedFrameMaps)
      delete map;
  for (auto map : detectedFrameMaps)
      delete map;
  if (reservedMem) cudaFreeHost(reservedMem);

#ifdef PROFILE
  // Kill profiler
  kill(pid,SIGINT);
  waitpid(pid,nullptr,0);
#endif
  //ecrypt "decryptedFile" and place back in originalFilename
  if(encryptFileCBC(originalFilename) == -1){
      LOG(ERROR) << "Error encrypting file.";
      return EXIT_FAILURE;
  }
  return 0;
}
