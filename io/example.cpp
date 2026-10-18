#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../myCamera.hpp"
int main() {
#if defined(__linux__)
if (getenv("DISPLAY")) setenv("GDK_BACKEND", "x11", 1); // Linux only: 让 GTK 窗口显示在 VMware 桌面
#endif
myCamera camera;
cv::Mat frame;
while (true) {
frame = camera.read();
if (frame.empty()) break;
std::cout << "frame " << frame.cols << "x" << frame.rows << std::endl;
cv::imshow("My Camera",frame);
if (cv::waitKey(30) == 'q')  break;
}
return 0;
}
