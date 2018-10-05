#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <deque>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <thread>
#include <sys/time.h>

#define FLATBUFFERS_DEBUG_VERIFICATION_FAILURE
#include "darknetserver.grpc.fb.h"

using grpc::Channel;
using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
using grpc::CompletionQueue;
using grpc::Status;
using darknetServer::DetectedObjects;
using darknetServer::KeyFrame;
using darknetServer::ImageDetection;

struct timestamp {
    struct timeval start;
    struct timeval end;
};

static inline void tvsub(struct timeval *x,
						 struct timeval *y,
						 struct timeval *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_usec = x->tv_usec - y->tv_usec;
	if (ret->tv_usec < 0) {
		ret->tv_sec--;
		ret->tv_usec += 1000000;
	}
}

void probe_time_start2(struct timestamp *ts)
{
    gettimeofday(&ts->start, NULL);
}

float probe_time_end2(struct timestamp *ts)
{
    struct timeval tv;
    gettimeofday(&ts->end, NULL);
	tvsub(&ts->end, &ts->start, &tv);
	return (tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0);
}

typedef struct {
	int width;
	int height;
	int numChannels;
	int widthStep;
	float *data;
} Image;

void printImage(Image &image)
{
	std::cout << "width: " << image.width;
	std::cout << " height: " << image.height <<std::endl;
	std::cout << "numChannels: " << image.numChannels <<std::endl;
	std::cout << "widthStep: " << image.widthStep <<std::endl;
	std::cout <<"Image Size:" << image.width*image.height*image.numChannels <<std::endl;
}

Image getImageFromMat(cv::Mat *m)
{
	Image image;
	image.height = m->rows;
	image.width = m->cols;
	image.numChannels = m->channels();
	image.widthStep = (int)m->step;
	image.data = new float[image.height*image.width*image.numChannels]();

	for(int h = 0; h < image.height; ++h){
		for(int c= 0; c < image.numChannels; ++c){
			for(int w = 0; w < image.width; ++w){
				image.data[(c*image.width*image.height + h*image.width + w)] = m->data[h*image.widthStep + w*image.numChannels + c]/255.0;
			}
		}
	}
	return image;
}

cv::Mat resizeKeepAspectRatio(const cv::Mat &input, const cv::Size &dstSize, const cv::Scalar &bgcolor)
{
	cv::Mat output;

	double h1 = dstSize.width * (input.rows/(double)input.cols);
	double w2 = dstSize.height * (input.cols/(double)input.rows);
	if( h1 <= dstSize.height) {
	    cv::resize( input, output, cv::Size(dstSize.width, h1));
	} else {
	    cv::resize( input, output, cv::Size(w2, dstSize.height));
	}

	int top = (dstSize.height-output.rows) / 2;
	int down = (dstSize.height-output.rows+1) / 2;
	int left = (dstSize.width - output.cols) / 2;
	int right = (dstSize.width - output.cols+1) / 2;

	cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor );

	return output;
}

void readFile(const std::string& filename, std::string& data)
{
	std::ifstream file(filename.c_str(), std::ios::in);

	if(file.is_open()) {
		std::stringstream ss;
		ss << file.rdbuf();
		file.close ();

		data = ss.str();
	}
	return;
}

class ImageDetectionClient {
  public:
	explicit ImageDetectionClient(std::shared_ptr<Channel> channel, int numThreads)
			: stub_(ImageDetection::NewStub(channel)) {}

	// Assembles the client's payload and sends it to the server.
	void AsyncSendImage(Image *image, std::function<void(void)> callback, flatbuffers::grpc::MessageBuilder *messageBuilder)
	{
		// Call object to store RPC data
		AsyncClientCall* call = new AsyncClientCall;

		// Start timer
		probe_time_start2(&call->ts_detect);

		call->completionCallback = callback;

		// Use the messageBuilder to construct a message from the image passed to us.
		auto requestOffset = darknetServer::CreateKeyFrame(*messageBuilder,
													image->width, image->height,
													image->numChannels, image->widthStep,
													messageBuilder->CreateVector<float>(image->data, image->height*image->width*image->numChannels));
		messageBuilder->Finish(requestOffset);
		// grab the message, so we are the owners.
		auto frameFBMessage = messageBuilder->ReleaseMessage<KeyFrame>();
		frameFBMessage.Verify();

		// Set a deadline of 200ms. We're not willing to wait more than that per frame
		//std::chrono::system_clock::time_point deadline =
		//	std::chrono::system_clock::now() + std::chrono::milliseconds(200);
		//call->context.set_deadline(deadline);

		// stub_->PrepareAsyncSayHello() creates an RPC object, returning
		// an instance to store in "call" but does not actually start the RPC
		// Because we are using the asynchronous API, we need to hold on to
		// the "call" instance in order to get updates on the ongoing RPC.
		call->async_reader =
			stub_->PrepareAsyncRequestDetection(&call->context, frameFBMessage, &cq_);

		// StartCall initiates the RPC call
		// Tag: the memory address of the call object.
		call->async_reader->StartCall();

		// Request that, upon completion of the RPC, "detectedObjectsFBMessage" be updated with the
		// server's response, and "status" with the indication of whether the operation
		// was successful. Tag the request with the memory address of the call object.
		call->async_reader->Finish(&call->detectedObjectsFBMessage, &call->status, (void*)call);
	}

	// Loop while listening for completed responses.
	// Prints out the response from the server.
	void AsyncCompleteRpc()
	{
		void* got_tag;
		bool ok = false;

		// Block until the next result is available in the completion queue "cq".
		while (cq_.Next(&got_tag, &ok)) {
			// Verify that Next() completed successfully.
			GPR_ASSERT(ok);

			AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);

			if (call->status.ok()) {
				// print out what we received...
				const DetectedObjects *detectedObjects = call->detectedObjectsFBMessage.GetRoot();
				std::cout << " " << detectedObjects->numObjects() << " objects detected."
						  <<std::endl;
			} else {
				std::cout << "RPC failed: " << call->status.error_code() <<": " <<call->status.error_message() << std::endl;
			}

			// Inform the client thread that this request is complete
			call->completionCallback();

			std::cout << "This request took " << probe_time_end2(&call->ts_detect) << " milliseconds"<< std::endl;

			// Once we're done, deallocate the call object.
			delete call;
		}
	}

  private:

	// struct for keeping state and data information
	struct AsyncClientCall {
		// Container for the data we expect from the server.
		flatbuffers::grpc::Message<DetectedObjects> detectedObjectsFBMessage;

		// Timestamps
		struct timestamp ts_detect;

		// Context for the client. It could be used to convey extra information to
		// the server and/or tweak certain RPC behaviors.
		ClientContext context;

		std::function<void(void)> completionCallback;

		// Storage for the status of the RPC upon completion.
		Status status;

		std::unique_ptr<grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<DetectedObjects>>> async_reader;
	};

	// Our view of the server's exposed services.
	std::unique_ptr<ImageDetection::Stub> stub_;

	// TODO: add deadlines to the tasks.
	// The producer-consumer queue we use to communicate asynchronously with the
	// gRPC runtime.
	CompletionQueue cq_;
};

class FrameMap {
public:

	void insert(Image &image, std::uint64_t frameNum)
	{
		std::lock_guard<std::mutex> lock(this->mutex);
		frames.emplace(std::make_pair(frameNum,image));
		cv.notify_all();
	}

	bool getImage(Image &image, std::uint64_t frameNum)
	{
		std::unique_lock<std::mutex> lock(this->mutex);
		if (frames.empty())
			cv.wait(lock, [this](){ return !this->frames.empty(); });

		auto iterator = frames.find(frameNum);
		if (iterator == frames.end()) {
			return false;
		} else {
			image = iterator->second;
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
	std::unordered_map<std::uint64_t, Image> frames;
	std::mutex mutex;
	std::condition_variable cv;

};

class RequestThread {
  public:
	void initAndStartRunning(ImageDetectionClient *detectionClient, FrameMap *frames, int fps, int maxOutstandingPerThread, int threadID)
	{
		this->detectionClient = detectionClient;
		this->frames = frames;
		this->fps = fps;
		this->maxOutstandingPerThread = maxOutstandingPerThread;
		this->numDropped = 0;
		this->currentFrame = 1;
		this->threadID = threadID;
		thread = std::thread(&RequestThread::makeRequests, this);
	}

	void makeRequests()
	{
		Image image;

		while (true) {
			bool gotImage = false;
			int i = 0;
			while (!gotImage) {
				gotImage = frames->getImage(image, currentFrame);
			}

			if (outstandingRequests.load(std::memory_order_acquire) > maxOutstandingPerThread) {
				numDropped++;
			} else {
				detectionClient->AsyncSendImage(&image, std::bind(&RequestThread::decrementOutstanding, this), &messageBuilder);
				outstandingRequests.fetch_add(1, std::memory_order_release);
			}
			currentFrame++;
			usleep(1000000/fps);
		}
	}

	void decrementOutstanding()
	{
		outstandingRequests.fetch_sub(1, std::memory_order_release);
	}

	std::uint64_t getCurrentFrame()
	{
		return currentFrame;
	}

	std::uint64_t getNumDropped()
	{
		return numDropped;
	}

  private:
	// Constants defined by user on command line
	int fps;
	int maxOutstandingPerThread;
	int threadID;

	// Variables we operate on
	std::uint64_t currentFrame;
	std::uint64_t numDropped;
	std::atomic<unsigned int> outstandingRequests;

	// Objects (or pointers to)
	std::thread thread;
	ImageDetectionClient *detectionClient;
	FrameMap *frames;

	// Not thread safe; 1 per thread so we don't end up in funky scenarios.
	flatbuffers::grpc::MessageBuilder messageBuilder;
};

void printUsage(int argc, char**argv)
{
	std::cout << "Usage:" << std::endl << argv[0] << " -v <vid_file> [-n number-of-clients(default=1; valid range: 1 to 12) -f fps (default=30fps; valid range: 1 to 120) -r per_client_max_outstanding_requests (default=90; valid range = 1 to 1000) ]" << std::endl;
}

int main(int argc, char** argv)
{
	// Check args
	char *filename;
	int numThreads = 1;
	int fps = 30;
	int maxOutstandingPerThread = 90;

	if (argc < 3 || 0==(argc%2)) {
		printUsage(argc, argv);
		return EXIT_FAILURE;
	}

	for(int i = 1; i < argc-1; i=i+2){
		if(0==strcmp(argv[i], "-v")){
			filename = argv[i+1];
		} else if (0 == strcmp(argv[i], "-n")){
			numThreads = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-f")) {
			fps = atoi(argv[i+1]);
		} else if (0 == strcmp(argv[i], "-r")) {
			maxOutstandingPerThread = atoi(argv[i+1]);
		}
	}

	if (NULL == filename) {
		std::cout << "Please specify input video file." << std::endl;
		printUsage(argc, argv);
		return EXIT_FAILURE;
	}

	if (numThreads > 12) {
		std::cout << "Max concurrent clients supported = 12. Setting numThreads to 12"	<<std::endl;
		numThreads = 12;
	}

	if (fps > 120) {
		std::cout << "Max FPS supported = 120. Setting fps to 120"	<<std::endl;
		numThreads = 120;
	}
	if (maxOutstandingPerThread > 1000) {
		std::cout << "Max outstanding requests per thread supported = 1000. Resetting to 90"	<<std::endl;
		maxOutstandingPerThread = 90;
	}

	std::cout << "video file:" << filename <<std::endl;
	std::cout << "Creating " << numThreads << "threads, each producing frames at " << fps << " FPS."  <<std::endl;
	std::cout << "Each thread can have a maximum of " << maxOutstandingPerThread << " outstanding requests at any time. All other frames will be dropped."<<std::endl;
	std::cout << "Press control-c to quit at any point" << std::endl;

	// Used to override default gRPC channel values.
	grpc::ChannelArguments ch_args;
	// Our images tend to be ~4MiB. gRPC's default MaxMessageSize is much smaller.
	ch_args.SetMaxReceiveMessageSize(INT_MAX);

	std::string key;
	std::string cert;
	std::string root;

	readFile("client.crt", cert);
	readFile("client.key", key);
	readFile("ca.crt", root);

	grpc::SslCredentialsOptions SslCredOpts = {root, key, cert};

	// Instantiate the client. It requires a channel, used to invoke the RPCs
	// This channel models a connection to an endpoint (in this case,
	// localhost at port 50051). We indicate that the channel isn't authenticated
	// (use of InsecureChannelCredentials()).
	ImageDetectionClient detectionClient(grpc::CreateCustomChannel(
			"zemaitis:50051", grpc::SslCredentials(SslCredOpts), ch_args), numThreads);

	// Map that stores the frames so that we aren't burning CPU decoding the CV stream
	FrameMap frames;

	// Spawn request and completion threads
	std::vector<RequestThread> requestThreads(numThreads);
	std::vector<std::thread> completionThreads(numThreads);
	for (int i = 0; i < numThreads; i++) {
		completionThreads[i] = std::thread(&ImageDetectionClient::AsyncCompleteRpc, &detectionClient);
		requestThreads[i].initAndStartRunning(&detectionClient, &frames, fps, maxOutstandingPerThread, i);
	}

	// Open the video file and read video frames.
	// TODO: We may want to move this over the server, and stream a video instead...
	cv::VideoCapture capture(filename);
	// Decode and obtain KeyFrames
	if (capture.isOpened()) {
		cv::Mat capturedFrameMat;
		// TODO: Catch overflow
		std::uint64_t frameNum = 1;
		std::uint64_t lastFrameDequeued = 0;

		// Put all of the images into memory so that the worker threads can pick them up
		while(capture.read(capturedFrameMat)) {
			// Resize image to 410x410
			cv::Mat resizedFrameMat = resizeKeepAspectRatio(capturedFrameMat, cv::Size(416, 416), cv::Scalar(0,0,0));
			// Convert the image from cv::Mat to the image format that darknet expects
			Image image = getImageFromMat(&resizedFrameMat);

			// Insert the image into the FrameArray for the client threads to pick up.
			frames.insert(image, frameNum++);

		}

		// Now try to clean up.
		while(true) {
			// Try to remove any processed frames every time we insert 100 frames.
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

			// Rate-limit so that we don't consume too much CPU...
			sleep(2);
		}
		for (auto &thread : completionThreads)
			thread.join();
	} else {
		std::cout << "Couldn't open " << filename <<std::endl;
		return -1;
	}

	return 0;
}
