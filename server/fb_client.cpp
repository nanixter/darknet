#include "opencv2/highgui/highgui.hpp"
#include <iostream>
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
	std::cout << "Data: " << std::endl;
	std::cout <<"Image Size:" << image.width*image.height*image.numChannels <<std::endl;
	//for (int i = 0; i < (image.width * image.height * image.numChannels); i++ )
	//	std::cout << image.data[i];
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
	// TODO: add deadlines to the tasks.
	void sendImage(Image *image) {
		// Man this library has such awkward interfaces!
		// Gotta use this builder object to build the messages.
		flatbuffers::grpc::MessageBuilder messageBuilder;
		// Container for the data we expect from the server.
		flatbuffers::grpc::Message<DetectedObjects> detectedObjectsFBMessage;
		// Context for the client. It could be used to convey extra information to
		// the server and/or tweak certain RPC behaviors.
		ClientContext context;
		// Storage for the status of the RPC upon completion.
		Status status;
		// Timestamps
		struct timestamp ts_detect;

		probe_time_start2(&ts_detect);

		//printImage(*image);
		// Use the messageBuilder to construct a message from the image passed to us.
		auto requestOffset = darknetServer::CreateKeyFrame(messageBuilder, 
													image->width, image->height,
													image->numChannels, image->widthStep,
													messageBuilder.CreateVector<float>(image->data, image->height*image->width*image->numChannels));
		messageBuilder.Finish(requestOffset);
		// grab the message, so we are the owners.
		auto frameFBMessage = messageBuilder.ReleaseMessage<KeyFrame>();
		frameFBMessage.Verify();

		// RPC Call
		status =  stub_->RequestDetection(&context, frameFBMessage, &detectedObjectsFBMessage);

		if (status.ok()) {
			const DetectedObjects *detectedObjects = detectedObjectsFBMessage.GetRoot();
			std::cout << " " << detectedObjects->numObjects()
						<< " objects detected." 	<<std::endl;
		} else {
			std::cout << "RPC failed: " << status.error_code() <<": "
						<<status.error_message() << std::endl;
		}

		// Print out the time it took to service the request.
		std::cout << "This request took " << probe_time_end2(&ts_detect)
					<< " milliseconds"<< std::endl;
	}

  private:
	// The stub is our view of the server's exposed services.
	std::unique_ptr<ImageDetection::Stub> stub_;
};

int main(int argc, char** argv) {
	// Used to override default gRPC channel values.
	grpc::ChannelArguments ch_args;
	// Our images tend to be ~4MiB. gRPC's default MaxMessageSize is much smaller.
	ch_args.SetMaxReceiveMessageSize(INT_MAX);

	// Instantiate the client. It requires a channel, used to invoke the RPCs
	// This channel models a connection to an endpoint (in this case,
	// localhost at port 50051). We indicate that the channel isn't authenticated
	// (use of InsecureChannelCredentials()).
	// TODO: Replace with an authenticated channel
	ImageDetectionClient detectionClient(grpc::CreateCustomChannel(
			"localhost:50051", grpc::InsecureChannelCredentials(), ch_args));

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
			// Convert the image from cv::Mat to the image format that darknet expects
			Image image = getImageFromMat(&capturedFrame);

			// The actual RPC call!
			detectionClient.sendImage(&image);

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
