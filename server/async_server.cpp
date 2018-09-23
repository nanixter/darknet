#include <iostream>
#include <fstream>
#include <sstream>
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

#define FLATBUFFERS_DEBUG_VERIFICATION_FAILURE
#include "darknetserver.grpc.fb.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerCompletionQueue;
using grpc::Status;
using darknetServer::DetectedObjects;
using darknetServer::DetectedObject;
using darknetServer::KeyFrame;
using darknetServer::bbox;
using darknetServer::ImageDetection;

#include "darknet_wrapper.h"

using DarknetWrapper::WorkRequest;
using DarknetWrapper::DetectionQueue;
using DarknetWrapper::AsyncDetector;


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
		std::string server_address("zemaitis:50051");
		//std::string server_address("128.83.122.71:50051");

		// Initialize detector - pass it the request and completion queues
		// Initialization must be done before launching the detection thread.
		detector.Init(argc, argv, &requestQueue, &completionQueue);

		// start a Thread to run doDetection
		std::thread detectionThread(&AsyncDetector::doDetection, &detector);

		ServerBuilder builder;
		std::string key;
		std::string cert;
		std::string root;

		this->readFile("server.crt", cert);
		this->readFile("server.key", key);
		this->readFile("ca.crt", root);

		grpc::SslServerCredentialsOptions::PemKeyCertPair keycert = {key, cert};

		grpc::SslServerCredentialsOptions sslOps;
		sslOps.pem_root_certs = root;
		sslOps.pem_key_cert_pairs.push_back (keycert);

		// Listen on the given address with TLS authentication.
		builder.AddListeningPort(server_address, grpc::SslServerCredentials( sslOps ));
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
		std::thread laterHalfThread(&ServerImpl::doLaterHalf, this);

		// The server's main loop.
		doFirstHalf();

		// Wait for the other threads to finish
		// What's the point of this. doFirstHalf never returns anyways...
		detectionThread.join();
		laterHalfThread.join();
	}

 private:
 	void readFile(const std::string& filename, std::string& data)
	{
		std::ifstream file(filename.c_str(), std::ifstream::in);

		if(file.is_open()) {
			std::stringstream ss;
			ss << file.rdbuf();
			file.close ();

			data = ss.str();
		}
		return;
	}

	// Class encompasing the state and logic needed to serve a request.
	class CallData {
	 public:
		// Take in the "service" instance (in this case representing an asynchronous
		// server) and the completion queue "cq" used for asynchronous communication
		// with the gRPC runtime.
		CallData(ImageDetection::AsyncService* service, ServerCompletionQueue* cq, DetectionQueue *requestQ, AsyncDetector *detector)
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
				service_->RequestRequestDetection(&ctx_, &requestMessage, &asyncResponder, cq_, cq_, this);
			} else if (status_ == READY) {
				probe_time_start2(&ts_server);
				// Spawn a new CallData instance to serve new clients while we process
				// the one for this CallData. The instance will deallocate itself as
				// part of its FINISH state.
				new CallData(service_, cq_, requestQueue, detector);

				// The actual processing.
				WorkRequest work;
				work.done = false;
				work.tag = this;
				work.img = detector->convertImage(requestMessage.GetRoot());
				work.dets = nullptr;
				work.nboxes = 0;

				requestQueue->push_back(work);
				status_ = PROCESSING;
			} else {
				GPR_ASSERT(status_ == FINISH);
				// Once in the FINISH state, deallocate ourselves (CallData).
				delete this;
			}
		}

		void completeRequest(WorkRequest &work) {
			GPR_ASSERT(status_ == PROCESSING);
			GPR_ASSERT(work.done == true);
			GPR_ASSERT(work.dets != nullptr);

			std::vector<flatbuffers::Offset<DetectedObject>> objects;
			int numObjects = 0;
			detection *dets = work.dets;
			for (int i = 0; i < work.nboxes; i++) {
				if(dets[i].objectness == 0) continue;
				bbox box(dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
				std::vector<float> prob;
				for (int j = 0; j < work.classes; j++) {
					prob.push_back(dets[i].prob[j]);
				}
				auto objectOffset = darknetServer::CreateDetectedObjectDirect(messageBuilder, &box, dets[i].classes, dets[i].objectness, dets[i].sort_class, &prob);
				objects.push_back(objectOffset);
				numObjects++;
			}

			flatbuffers::Offset<DetectedObjects> detectedObjectsOffset = darknetServer::CreateDetectedObjectsDirect(messageBuilder, numObjects, &objects);

			messageBuilder.Finish(detectedObjectsOffset);
			this->responseMessage = messageBuilder.ReleaseMessage<DetectedObjects>();
			GPR_ASSERT(this->responseMessage.Verify());

			// Clean up
			free_detections(work.dets, work.nboxes);

			status_ = FINISH;
			std::cout << "Total server time for this frame: " << probe_time_end2(&ts_server) << " milliseconds"<< std::endl;
			asyncResponder.Finish(this->responseMessage, Status::OK, this);
		}

	 private:
		//Total query time from the time we recieve the packet.
		struct timestamp ts_server;

		// The means of communication with the gRPC runtime for an asynchronous
		// server.
		ImageDetection::AsyncService* service_;
		// The producer-consumer queue where for asynchronous server notifications.
		ServerCompletionQueue* cq_;
		// Context for the rpc, allowing to tweak aspects of it such as the use
		// of compression, authentication, as well as to send metadata back to the
		// client.
		ServerContext ctx_;

		DetectionQueue *requestQueue;
		AsyncDetector *detector;

		// Used to make Flatbuffer messages...
		flatbuffers::grpc::MessageBuilder messageBuilder;

		// What we get from the client.
		flatbuffers::grpc::Message<KeyFrame> requestMessage;

		flatbuffers::grpc::Message<DetectedObjects> responseMessage;

		// The means to get back to the client.
		ServerAsyncResponseWriter<flatbuffers::grpc::Message<DetectedObjects>> asyncResponder;

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
			GPR_ASSERT(cq_->Next(&tag, &ok));
			GPR_ASSERT(ok);
			static_cast<CallData*>(tag)->scheduleRequest();
		}
	}

	void doLaterHalf() {
		while(true) {
			WorkRequest work;
			completionQueue.pop_front(work);
			static_cast<CallData*>(work.tag)->completeRequest(work);
		}
	}

	// Darknet detector
	AsyncDetector detector;
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
