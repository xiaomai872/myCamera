#ifndef MYCOLORCORRECT_HPP
#define MYCOLORCORRECT_HPP

#include <string>
#include <vector>
#include <opencv2/core.hpp>

class myColorCorrect
{
public:
    myColorCorrect();
    ~myColorCorrect();

    cv::Mat read();
private:
    void _estimateBandGains(const cv::Mat& bgr,
                            std::vector<double>& kR,
                            std::vector<double>& kG,
                            std::vector<double>& kB) const;

    cv::Mat _applyBandGains(const cv::Mat& bgr,
                            const std::vector<double>& kR,
                            const std::vector<double>& kG,
                            const std::vector<double>& kB,
                            double w) const;

    cv::Mat _segmentWhiteBalance(const cv::Mat& bgr, double w) const;

    cv::Mat _chromaDenoise(const cv::Mat& bgr) const;

    cv::Mat _hueSelectiveSaturation(const cv::Mat& bgr, double satMax) const;

    void _neutralizeCast(cv::Mat& bgr, double strength) const;

    cv::Mat _protect(const cv::Mat& bgr, double satMax, bool warm, bool sharpen) const;

    bool _saveOutputs() const;
    void _reportMetrics(const cv::Mat& image, const std::string& tag) const;
    void _reportParams() const;
    double _highFreqStd(const cv::Mat& image, int channel) const;

    cv::Mat src_;
    cv::Mat wb_;
    cv::Mat dst_;

    std::string inputPath_;
    std::string outputDir_;
    std::string expDir_;

    int    bands_;
    double trimRatio_;
    int    smoothWin_;

    double scStrength_;
    double hlKnee_;
    double hlComp_;
    double unsharpAmount_;
    double unsharpSigma_;
    int    bilatD_;
    double bilatSigmaColor_; // bilateralFilter sigmaColor
    double bilatSigmaSpace_; // bilateralFilter sigmaSpace

    int    chromaMedK_;
    int    chromaBilD_;
    double chromaBilSigmaColor_;
    double chromaBilSigmaSpace_;

    double hueSatMax_;
    double hueLo_;
    double hueSpan_;
    double satLo_;
    double satSpan_;
    double gainSmoothSigma_;
    double chromaPostSigma_;

    double warmRScale_;
    double warmBScale_;
};

#endif // MYCOLORCORRECT_HPP
