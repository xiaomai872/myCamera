#include <opencv2/opencv.hpp>
class myCamera {
public:
myCamera();
~myCamera();
cv::Mat read();

private:
cv::VideoCapture capture_;
int index_;
};
