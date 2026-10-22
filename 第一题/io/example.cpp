#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../myCamera.hpp"
int main() {
#if defined(__linux__)
if (getenv("DISPLAY")) setenv("GDK_BACKEND", "x11", 1);
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
