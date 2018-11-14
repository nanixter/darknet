#ifndef TYPES_H
#define TYPES_H

#include "../../include/darknet.h"

namespace LiveStreamDetector {

	typedef struct {
		uint8_t *data = nullptr;
		int frameSize = 0;
		uint64_t frameNum;
		Timer timer;
		void *decompressedFrameRGBDevice = nullptr;
		void *decompressedFrameDevice = nullptr;
		bool finished = 0;
		int deviceNum;
		int streamNum;
	} Frame;

	typedef struct
	{
		bool done;
		bool finished;
		image img;
		detection *dets;
		int nboxes;
		int classes;
		Frame *tag;
	} WorkRequest;
} //namespace

#endif //TYPES_H
