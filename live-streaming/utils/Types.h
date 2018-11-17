#ifndef TYPES_H
#define TYPES_H

#include "../../include/darknet.h"

namespace LiveStreamDetector {

	typedef struct {
		uint8_t *data = nullptr;
		int frameSize = 0;
		uint64_t frameNum;
		Timer timer;
		void *decompressedFrameDevice = nullptr;
		int deviceNumDecompressed;
		int decompressedFrameSize;
		void *decompressedFrameRGBDevice = nullptr;
		int decompressedFrameRGBSize;
		int deviceNumRGB;
		bool finished = 0;
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
		int deviceNum;
		Frame *tag;
	} WorkRequest;
} //namespace

#endif //TYPES_H
