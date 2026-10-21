#include "myCamera.hpp"
myCamera::myCamera() {
	// Cross-platform capture backend
#ifdef _WIN32
	capture_.open(0, cv::CAP_DSHOW);      // Windows: DirectShow
#else
	capture_.open(0, cv::CAP_V4L2);       // Linux: Video4Linux2
#endif
	if (capture_.isOpened()) {
		capture_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
		capture_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		capture_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);
		capture_.set(cv::CAP_PROP_BRIGHTNESS, 135);
		capture_.set(cv::CAP_PROP_GAMMA, 118);
		capture_.set(cv::CAP_PROP_CONTRAST, 40);
		capture_.set(cv::CAP_PROP_SATURATION, 64);
		capture_.set(cv::CAP_PROP_AUTO_WB, 0);
		capture_.set(cv::CAP_PROP_WB_TEMPERATURE, 4800);
		std::cout << "ctrl readback: auto_exposure=" << capture_.get(cv::CAP_PROP_AUTO_EXPOSURE)
		          << " brightness=" << capture_.get(cv::CAP_PROP_BRIGHTNESS)
		          << " gamma=" << capture_.get(cv::CAP_PROP_GAMMA)
		          << " contrast=" << capture_.get(cv::CAP_PROP_CONTRAST)
		          << " saturation=" << capture_.get(cv::CAP_PROP_SATURATION)
		          << " auto_wb=" << capture_.get(cv::CAP_PROP_AUTO_WB)
		          << " wb_temp=" << capture_.get(cv::CAP_PROP_WB_TEMPERATURE)
		          << std::endl;
		cv::Mat warmup;
		for (int i = 0; i < 120; ++i) capture_ >> warmup;
	}
}
myCamera::~myCamera() {
	capture_.release();
}
cv::Mat myCamera::read() {
	cv::Mat frame;
	capture_ >> frame;
	return frame;
}
