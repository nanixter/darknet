#include "darknet.h"

void test_server_detection(const char *videoFile) {
	std::cout << "TEST" <<std::endl;
	char *videoFile = argv[4];
	CvCapture * cap; = cvCreateFileCapture(videoFile);

	image newImage;
	image newImage_letterboxed;
	detection *dets = nullptr;
	int nboxes = 0;

	int status = fill_image_from_stream(cap, newImage);
	while (status != 0) {
		letterbox_image_into(newImage, net->w, net->h, newImage_letterboxed);

		/* Now we finally run the actual network	*/
		network_predict(net, newImage_letterboxed.data);
		this->remember_network();
		dets = this->average_predictions(&nboxes, newImage.h, newImage.w);

		std::cout << "nboxes = " << nboxes << std::endl;
		status = fill_image_from_stream(cap, newImage);
	}
}