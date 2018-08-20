#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cstdio>

#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <thread>
#include <sys/time.h>

#include "darknetserver.grpc.pb.h"

#include "darknet_wrapper.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using darknetServer::DetectedObjects;
using darknetServer::KeyFrame;
using darknetServer::ImageDetection;
using DarknetWrapper::WorkRequest;
using DarknetWrapper::DetectionQueue;
using DarknetWrapper::Detector;

class ServerImpl final {
 public:
	~ServerImpl() {
		detector.Shutdown();
		server_->Shutdown();
		// Always shutdown the completion queue after the server.
		cq_->Shutdown();
	}

	// There is no shutdown handling in this code.
	void Run(int argc, char** argv) {
		std::string server_address("localhost:50051");
		//std::string server_address("128.83.122.71:50051");

		// Initialize detector - pass it the request and completion queues
		// Initialization must be done before launching the detection thread.
		detector.Init(argc, argv, &requestQueue, &completionQueue);

		// start a Thread to run doDetection
		//std::thread detectionThread(&Detector::doDetection, &detector);

		ServerBuilder builder;
		// Listen on the given address without any authentication mechanism.
		builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
		builder.SetMaxReceiveMessageSize(INT_MAX);
		// Register "service_" as the instance through which we'll communicate with
		// clients. In this case it corresponds to an *asynchronous* service.
		builder.RegisterService(&service);
		// Get hold of the completion queue used for the asynchronous communication
		// with the gRPC runtime.
		cq_ = builder.AddCompletionQueue();
		// Finally assemble the server.
		server_ = builder.BuildAndStart();
		std::cout << "Server listening on " << server_address << std::endl;

		// Start the thread that handles the second half of the processing.
		//std::thread laterHalfThread(&ServerImpl::doLaterHalf, this);
		// The server's main loop.
		doFirstHalf();

		// Wait for the other threads to
		//detectionThread.join();
		//laterHalfThread.join();
	}

 private:
	// Class encompasing the state and logic needed to serve a request.
	class CallData {
	 public:
		// Take in the "service" instance (in this case representing an asynchronous
		// server) and the completion queue "cq" used for asynchronous communication
		// with the gRPC runtime.
		CallData(ImageDetection::AsyncService* service, ServerCompletionQueue* cq, DetectionQueue *requestQ, Detector *detector)
				: service_(service), cq_(cq), asyncResponder(&ctx_), status_(CREATE) {
			// Invoke the serving logic right away.
			this->requestQueue = requestQ;
			this->detector = detector;
			scheduleRequest();
		}

		void scheduleRequest() {
			if (status_ == CREATE) {
				// Make this instance progress to the PROCESS state.
				status_ = READY;

				// As part of the initial CREATE state, we *request* that the system
				// start processing RequestDetection requests. In this request, "this" acts as
				// the tag uniquely identifying the request (so that different CallData
				// instances can serve different requests concurrently).
				//std::cout << "New CallData " << this << " spawned" << std::endl;
				service_->RequestRequestDetection(&ctx_, &frame, &asyncResponder, cq_, cq_, this);
			} else if (status_ == READY) {
				// Spawn a new CallData instance to serve new clients while we process
				// the one for this CallData. The instance will deallocate itself as
				// part of its FINISH state.
				probe_time_start2(&ts_server);
				new CallData(service_, cq_, requestQueue, detector);

				// The actual processing.
				WorkRequest work;
				work.done = false;
				work.tag = this;
				work.frame = this->frame;
				//std::cout << "Recieved request " << this << " Pushing onto requestQ" << std::endl;
				//requestQueue->push_back(work);
				detector->doDetection(work);
				status_ = PROCESSING;
				completeRequest(work);
			} else {
				GPR_ASSERT(status_ == FINISH);
				// Once in the FINISH state, deallocate ourselves (CallData).
				delete this;
			}
		}

		void completeRequest(WorkRequest &work) {
				GPR_ASSERT(status_ == PROCESSING);
				GPR_ASSERT(work.done == true);
				//std::cout << "Request " << this << " completed." << std::endl;
				// GPU processing is done! Time to pass the results back to the client.
				this->objects = work.detectedObjects;
				status_ = FINISH;
				std::cout << work.tag << "Server took " << probe_time_end2(&ts_server) << " milliseconds"<< std::endl;
				asyncResponder.Finish((this->objects), Status::OK, this);
		}

	 private:
		// The means of communication with the gRPC runtime for an asynchronous
		// server.
		ImageDetection::AsyncService* service_;
		// The producer-consumer queue where for asynchronous server notifications.
		ServerCompletionQueue* cq_;
		// Context for the rpc, allowing to tweak aspects of it such as the use
		// of compression, authentication, as well as to send metadata back to the
		// client.
		ServerContext ctx_;

		struct timestamp ts_server;
		Detector *detector;
		DetectionQueue *requestQueue;

		// What we get from the client.
		KeyFrame frame;

		// What we send back to the client.
		DetectedObjects objects;

		// The means to get back to the client.
		ServerAsyncResponseWriter<DetectedObjects> asyncResponder;

		// Let's implement a tiny state machine with the following states.
		enum CallStatus { CREATE, READY, PROCESSING, FINISH };
		CallStatus status_;  // The current serving state.
	};

	// This can be run in multiple threads if needed.
	void doFirstHalf() {
		// Spawn a new CallData instance to serve new clients.
		new CallData(&service, cq_.get(), &requestQueue, &detector);
		void* tag;  // uniquely identifies a request.
		bool ok;
		while (true) {
			// Block waiting to read the next event from the completion queue. The
			// event is uniquely identified by its tag, which in this case is the
			// memory address of a CallData instance.
			// The return value of Next should always be checked. This return value
			// tells us whether there is any kind of event or cq_ is shutting down.
			//std::cout << __LINE__ << "sleep on grpc q" <<std::endl;
			GPR_ASSERT(cq_->Next(&tag, &ok));
			//std::cout << __LINE__ << "got req. Call schedule" <<std::endl;
			GPR_ASSERT(ok);
			static_cast<CallData*>(tag)->scheduleRequest();
		}
	}

	void doLaterHalf() {
		while(true) {
			WorkRequest work;
			//std::cout << __LINE__ << "doLaterHalf: sleep on queue" <<std::endl;
			completionQueue.pop_front(work);
			//std::cout << __LINE__ << "doLaterHalf: got completion notice" <<std::endl;
			static_cast<CallData*>(work.tag)->completeRequest(work);
		}
	}

	// Darknet detector
	Detector detector;
	DetectionQueue requestQueue;
	DetectionQueue completionQueue;

	std::unique_ptr<ServerCompletionQueue> cq_;
	ImageDetection::AsyncService service;
	std::unique_ptr<Server> server_;
};

int main(int argc, char** argv) {

	if(argc < 4){
		fprintf(stderr, "usage: %s <datacfg> <cfg> <weights>\n", argv[0]);
		return EXIT_FAILURE;
	}

	ServerImpl server;
	server.Run(argc, argv);

	return EXIT_SUCCESS;
}
