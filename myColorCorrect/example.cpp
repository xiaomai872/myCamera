#include <iostream>

#include "myColorCorrect.hpp"

int main()
{
    std::cout << "===== Task2 myColorCorrect (Ubuntu, system OpenCV) =====" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    myColorCorrect corrector;
    cv::Mat result = corrector.read();
    if (result.empty()) {
        std::cout << "RESULT: FAIL" << std::endl;
        return 1;
    }
    std::cout << "RESULT: PASS  corrected=" << result.cols << "x" << result.rows << std::endl;
    return 0;
}