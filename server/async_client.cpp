#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
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

void printImage(Image &image){
	std::cout << "width: " << image.width;
	std::cout << " height: " << image.height <<std::endl;
	std::cout << "numChannels: " << image.numChannels <<std::endl;
	std::cout << "widthStep: " << image.widthStep <<std::endl;
	std::cout <<"Image Size:" << image.width*image.height*image.numChannels <<std::endl;
}

Image getImageFromMat(cv::Mat *m) {
	Image image;
	image.height = m->rows;
	image.width = m->cols;
	image.numChannels = m->channels();
	image.widthStep = (int)m->step;
	image.data = new float[image.height*image.width*image.numChannels]();

	for(int i = 0; i < image.height; ++i){
		for(int k= 0; k < image.numChannels; ++k){
			for(int j = 0; j < image.width; ++j){
				image.data[(k*image.width*image.height + i*image.width + j)] = m->data[i*image.widthStep + j*image.numChannels + k]/255.0;
			}
		}
	}
	return image;
}

class ImageDetectionClient {
  public:
	explicit ImageDetectionClient(std::shared_ptr<Channel> channel)
			: stub_(ImageDetection::NewStub(channel)) {}

	// Assembles the client's payload and sends it to the server.
	void AsyncSendImage(Image *image) {
		// Call object to store RPC data
		AsyncClientCall* call = new AsyncClientCall;

		// Start timer
		probe_time_start2(&call->ts_detect);

		// Use the messageBuilder to construct a message from the image passed to us.
		auto requestOffset = darknetServer::CreateKeyFrame(this->messageBuilder,
													image->width, image->height,
													image->numChannels, image->widthStep,
													messageBuilder.CreateVector<float>(image->data, image->height*image->width*image->numChannels));
		messageBuilder.Finish(requestOffset);
		// grab the message, so we are the owners.
		auto frameFBMessage = messageBuilder.ReleaseMessage<KeyFrame>();
		frameFBMessage.Verify();

		// stub_->PrepareAsyncSayHello() creates an RPC object, returning
		// an instance to store in "call" but does not actually start the RPC
		// Because we are using the asynchronous API, we need to hold on to
		// the "call" instance in order to get updates on the ongoing RPC.
		call->async_reader =
			stub_->PrepareAsyncRequestDetection(&call->context, frameFBMessage, &cq_);

		// StartCall initiates the RPC call
		// Tag: the memory address of the call object.
		call->async_reader->StartCall();

		// Request that, upon completion of the RPC, "reply" be updated with the
		// server's response; "status" with the indication of whether the operation
		// was successful. Tag the request with the memory address of the call object.
		call->async_reader->Finish(&call->detectedObjectsFBMessage, &call->status, (void*)call);

	}

	// Loop while listening for completed responses.
	// Prints out the response from the server.
	void AsyncCompleteRpc() {
		void* got_tag;
		bool ok = false;

		// Block until the next result is available in the completion queue "cq".
		while (cq_.Next(&got_tag, &ok)) {
			// The tag in this example is the memory location of the call object
			AsyncClientCall* call = static_cast<AsyncClientCall*>(got_tag);

			// Store the completion time...
			//call->endTime = rdtsc();

			// Verify that the request was completed successfully. Note that "ok"
			// corresponds solely to the request for updates introduced by Finish().
			GPR_ASSERT(ok);


			if (call->status.ok()) {
				// print out what we received...
				const DetectedObjects *detectedObjects = call->detectedObjectsFBMessage.GetRoot();
				std::cout << " " << detectedObjects->numObjects() << " objects detected."
						  <<std::endl;
			} else {
				std::cout << "RPC failed: " << call->status.error_code() <<": " <<call->status.error_message() << std::endl;
			}

			std::cout << "This request took " << probe_time_end2(&call->ts_detect) << " milliseconds"<< std::endl;
			// Once we're complete, deallocate the call object.
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

		// Storage for the status of the RPC upon completion.
		Status status;

		std::unique_ptr<grpc::ClientAsyncResponseReader<flatbuffers::grpc::Message<DetectedObjects>>> async_reader;
	};

	// Gotta use this builder object to build the messages.
	flatbuffers::grpc::MessageBuilder messageBuilder;

	// Our view of the server's exposed services.
	std::unique_ptr<ImageDetection::Stub> stub_;

	// TODO: add deadlines to the tasks.
	// The producer-consumer queue we use to communicate asynchronously with the
	// gRPC runtime.
	CompletionQueue cq_;
};

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

int main(int argc, char** argv) {
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
			"zemaitis:50051", grpc::SslCredentials(SslCredOpts), ch_args));
			//"localhost:50051", grpc::InsecureChannelCredentials(), ch_args));

	// Spawn completion thread that loops indefinitely
	std::thread completionThread = std::thread(&ImageDetectionClient::AsyncCompleteRpc,
		&detectionClient);


	// Open the input video file
	// TODO: Fork multiple processes and send multiple video streams.
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
	printf("video file: %s\n", filename);
	std::cout << "Press control-c to quit at any point" << std::endl;

	// Open the video file and read video frames.
	// TODO: We may want to move this over the server, and stream a video instead...
	cv::VideoCapture capture(filename);
	// Decode and obtain KeyFrames
	if (capture.isOpened()) {
		cv::Mat capturedFrame;

		while(capture.read(capturedFrame)) {
			// Resize image to 410x410
			cv::Mat resizedFrame = resizeKeepAspectRatio(capturedFrame, cv::Size(416, 416), cv::Scalar(0,0,0));
			// Convert the image from cv::Mat to the image format that darknet expects
			Image image = getImageFromMat(&resizedFrame);

			// The actual RPC call!
			detectionClient.AsyncSendImage(&image);
			// Rate-limit ourselves.
			// For testing purposes only.
			sleep(1);
		}
	} else {
		std::cout << "Couldn't open " << filename <<std::endl;
		return -1;
	}

	return 0;
}
