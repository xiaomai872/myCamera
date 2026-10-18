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
		// 自动曝光保持 Aperture Priority Mode（该 UVC 手动曝光模式恒全黑）
		capture_.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);
		// 成像参数回调至中性区间：此前 brightness=200 / gamma=150 导致过曝发白
		capture_.set(cv::CAP_PROP_BRIGHTNESS, 135);   // 0~255，默认 128
		capture_.set(cv::CAP_PROP_GAMMA, 118);        // 90~150，默认 120
		capture_.set(cv::CAP_PROP_CONTRAST, 40);      // 0~100，默认 32；32 偏低使画面发白、层次扁平
		capture_.set(cv::CAP_PROP_SATURATION, 64);    // 0~100，默认 64
		// 自动白平衡下 G 通道偏高、画面发绿，改为固定中性色温
		capture_.set(cv::CAP_PROP_AUTO_WB, 0);
		capture_.set(cv::CAP_PROP_WB_TEMPERATURE, 4800); // 2800~6500，实测 4800 最中性
		std::cout << "ctrl readback: auto_exposure=" << capture_.get(cv::CAP_PROP_AUTO_EXPOSURE)
		          << " brightness=" << capture_.get(cv::CAP_PROP_BRIGHTNESS)
		          << " gamma=" << capture_.get(cv::CAP_PROP_GAMMA)
		          << " contrast=" << capture_.get(cv::CAP_PROP_CONTRAST)
		          << " saturation=" << capture_.get(cv::CAP_PROP_SATURATION)
		          << " auto_wb=" << capture_.get(cv::CAP_PROP_AUTO_WB)
		          << " wb_temp=" << capture_.get(cv::CAP_PROP_WB_TEMPERATURE)
		          << std::endl;
		// 丢弃开流初期自动曝光尚未收敛的黑帧
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
