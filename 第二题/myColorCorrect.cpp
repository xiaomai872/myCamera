#include "myColorCorrect.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sys/stat.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

const double kLumB = 0.114;
const double kLumG = 0.587;
const double kLumR = 0.299;

inline double lumOf(double b, double g, double r)
{
    return kLumB * b + kLumG * g + kLumR * r;
}

inline double clamp255(double v)
{
    if (v < 0.0) return 0.0;
    if (v > 255.0) return 255.0;
    return v;
}

inline double clampd(double v, double lo, double hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline unsigned char toU8(double v)
{
    if (v < 0.0) return (unsigned char)0;
    if (v > 255.0) return (unsigned char)255;
    return (unsigned char)v;
}

bool dirExists(const std::string& path)
{
    struct stat st;
    if (::stat(path.c_str(), &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

} // namespace

myColorCorrect::myColorCorrect()
    : inputPath_("data/input.jpg"),
      outputDir_("output"),
      expDir_(std::string(getenv("HOME") ? getenv("HOME") : ".") + "/exp_wb"),
      bands_(16),
      trimRatio_(0.10),
      smoothWin_(5),
      scStrength_(0.10),
      hlKnee_(235.0),
      hlComp_(0.4),
      unsharpAmount_(0.25),
      unsharpSigma_(1.0),
      bilatD_(9),
      bilatSigmaColor_(40.0),
      bilatSigmaSpace_(9.0),
      chromaMedK_(3),
      chromaBilD_(5),
      chromaBilSigmaColor_(12.0),
      chromaBilSigmaSpace_(5.0),
      hueSatMax_(2.80),
      hueLo_(25.0),
      hueSpan_(40.0),
      satLo_(10.0),
      satSpan_(20.0),
      gainSmoothSigma_(4.0),
      chromaPostSigma_(1.0),
      warmRScale_(1.02),
      warmBScale_(0.98)
{
}

myColorCorrect::~myColorCorrect()
{
}
cv::Mat myColorCorrect::read()
{
    src_ = cv::imread(inputPath_, cv::IMREAD_COLOR);
    if (src_.empty()) {
        std::cerr << "[ERR] cannot read " << inputPath_ << std::endl;
        return cv::Mat();
    }

    _reportParams();
    _reportMetrics(src_, "input");

    wb_ = _segmentWhiteBalance(src_, 1.0);
    _reportMetrics(wb_, "wb(16band x2)");

    dst_ = _protect(wb_, hueSatMax_, true, true);

    if (!_saveOutputs()) {
        return cv::Mat();
    }
    _reportMetrics(dst_, "final");
    return dst_;
}

void myColorCorrect::_estimateBandGains(const cv::Mat& bgr,
                                       std::vector<double>& kR,
                                       std::vector<double>& kG,
                                       std::vector<double>& kB) const
{
    const int bands = bands_;
    const int rows = bgr.rows;
    const int cols = bgr.cols;

    std::vector<std::vector<float> > segL(bands);
    for (int y = 0; y < rows; ++y) {
        const cv::Vec3b* p = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; ++x) {
            const double b = p[x][0];
            const double g = p[x][1];
            const double r = p[x][2];
            int s = (int)(lumOf(b, g, r) * bands / 256.0);
            if (s < 0) s = 0;
            if (s >= bands) s = bands - 1;
            segL[s].push_back((float)lumOf(b, g, r));
        }
    }

    std::vector<double> lo(bands, 0.0);
    std::vector<double> hi(bands, 255.0);
    for (int s = 0; s < bands; ++s) {
        if (segL[s].size() < 100) continue;
        std::sort(segL[s].begin(), segL[s].end());
        const size_t n = segL[s].size();
        size_t iLo = (size_t)(trimRatio_ * (double)n);
        size_t iHi = (size_t)((1.0 - trimRatio_) * (double)n);
        if (iHi >= n) iHi = n - 1;
        lo[s] = (double)segL[s][iLo];
        hi[s] = (double)segL[s][iHi];
    }

    std::vector<double> sumR(bands, 0.0), sumG(bands, 0.0), sumB(bands, 0.0), sumL(bands, 0.0);
    std::vector<long long> cnt(bands, 0);
    for (int y = 0; y < rows; ++y) {
        const cv::Vec3b* p = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; ++x) {
            const double b = p[x][0];
            const double g = p[x][1];
            const double r = p[x][2];
            const double L = lumOf(b, g, r);
            int s = (int)(L * bands / 256.0);
            if (s < 0) s = 0;
            if (s >= bands) s = bands - 1;
            if (L < lo[s] || L > hi[s]) continue;
            sumR[s] += r;
            sumG[s] += g;
            sumB[s] += b;
            sumL[s] += L;
            cnt[s]++;
        }
    }

    kR.assign(bands, 1.0);
    kG.assign(bands, 1.0);
    kB.assign(bands, 1.0);
    for (int s = 0; s < bands; ++s) {
        if (cnt[s] < 32) continue;
        const double mr = sumR[s] / (double)cnt[s];
        const double mg = sumG[s] / (double)cnt[s];
        const double mb = sumB[s] / (double)cnt[s];
        const double ml = sumL[s] / (double)cnt[s];
        if (mr < 1e-6 || mg < 1e-6 || mb < 1e-6) continue;
        kR[s] = ml / mr;
        kG[s] = ml / mg;
        kB[s] = ml / mb;
    }

    const int half = smoothWin_ / 2;
    std::vector<double> sR = kR, sG = kG, sB = kB;
    for (int s = 0; s < bands; ++s) {
        const int a = std::max(0, s - half);
        const int z = std::min(bands, s + half + 1);
        double ar = 0.0, ag = 0.0, ab = 0.0;
        for (int i = a; i < z; ++i) {
            ar += kR[i];
            ag += kG[i];
            ab += kB[i];
        }
        const double n = (double)(z - a);
        sR[s] = ar / n;
        sG[s] = ag / n;
        sB[s] = ab / n;
    }
    kR.swap(sR);
    kG.swap(sG);
    kB.swap(sB);
}
cv::Mat myColorCorrect::_applyBandGains(const cv::Mat& bgr,
                                        const std::vector<double>& kR,
                                        const std::vector<double>& kG,
                                        const std::vector<double>& kB,
                                        double w) const
{
    const int bands = bands_;
    const int rows = bgr.rows;
    const int cols = bgr.cols;
    cv::Mat out(rows, cols, CV_32FC3);
    for (int y = 0; y < rows; ++y) {
        const cv::Vec3b* p = bgr.ptr<cv::Vec3b>(y);
        cv::Vec3f* q = out.ptr<cv::Vec3f>(y);
        for (int x = 0; x < cols; ++x) {
            const double b = p[x][0];
            const double g = p[x][1];
            const double r = p[x][2];
            const double L = lumOf(b, g, r);
            double pos = L * bands / 256.0 - 0.5;
            pos = clampd(pos, 0.0, (double)(bands - 1));
            int i0 = (int)std::floor(pos);
            if (i0 > bands - 1) i0 = bands - 1;
            int i1 = i0 + 1;
            if (i1 > bands - 1) i1 = bands - 1;
            const double t = pos - (double)i0;
            const double gr = 1.0 + w * ((kR[i0] * (1.0 - t) + kR[i1] * t) - 1.0);
            const double gg = 1.0 + w * ((kG[i0] * (1.0 - t) + kG[i1] * t) - 1.0);
            const double gb = 1.0 + w * ((kB[i0] * (1.0 - t) + kB[i1] * t) - 1.0);
            q[x][0] = (float)clamp255(b * gb);
            q[x][1] = (float)clamp255(g * gg);
            q[x][2] = (float)clamp255(r * gr);
        }
    }
    cv::Mat out8;
    out.convertTo(out8, CV_8U);
    return out8;
}

cv::Mat myColorCorrect::_segmentWhiteBalance(const cv::Mat& bgr, double w) const
{
    std::vector<double> kR, kG, kB;
    _estimateBandGains(bgr, kR, kG, kB);
    cv::Mat pass1 = _applyBandGains(bgr, kR, kG, kB, w);

    std::vector<double> kR2, kG2, kB2;
    _estimateBandGains(pass1, kR2, kG2, kB2);
    return _applyBandGains(pass1, kR2, kG2, kB2, w);
}

cv::Mat myColorCorrect::_chromaDenoise(const cv::Mat& bgr) const
{
    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ch;
    cv::split(ycc, ch);

    cv::Mat t;
    cv::medianBlur(ch[1], t, chromaMedK_); ch[1] = t.clone();
    cv::medianBlur(ch[2], t, chromaMedK_); ch[2] = t.clone();
    cv::bilateralFilter(ch[1], t, chromaBilD_, chromaBilSigmaColor_, chromaBilSigmaSpace_); ch[1] = t.clone();
    cv::bilateralFilter(ch[2], t, chromaBilD_, chromaBilSigmaColor_, chromaBilSigmaSpace_); ch[2] = t.clone();

    cv::merge(ch, ycc);
    cv::Mat out;
    cv::cvtColor(ycc, out, cv::COLOR_YCrCb2BGR);
    return out;
}


cv::Mat myColorCorrect::_hueSelectiveSaturation(const cv::Mat& bgr, double satMax) const
{
    cv::Mat sm;
    cv::bilateralFilter(bgr, sm, bilatD_, bilatSigmaColor_, bilatSigmaSpace_);
    cv::Mat hsvSm;
    cv::cvtColor(sm, hsvSm, cv::COLOR_BGR2HSV);

    cv::Mat gain(hsvSm.size(), CV_32F);
    for (int y = 0; y < hsvSm.rows; ++y) {
        const cv::Vec3b* p = hsvSm.ptr<cv::Vec3b>(y);
        float* g = gain.ptr<float>(y);
        for (int x = 0; x < hsvSm.cols; ++x) {
            const double h = (double)p[x][0] * 2.0;
            const double s = (double)p[x][1];
            double d = std::fabs(h - 120.0);
            if (d > 180.0) d = 360.0 - d;
            double wHue = (d - hueLo_) / hueSpan_;
            wHue = wHue < 0.0 ? 0.0 : (wHue > 1.0 ? 1.0 : wHue);
            double wSat = (s - satLo_) / satSpan_;
            wSat = wSat < 0.0 ? 0.0 : (wSat > 1.0 ? 1.0 : wSat);
            g[x] = (float)(1.0 + wHue * wSat * (satMax - 1.0));
        }
    }
    cv::Mat gSm;
    cv::GaussianBlur(gain, gSm, cv::Size(0, 0), gainSmoothSigma_);

    cv::Mat ycc;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> ch;
    cv::split(ycc, ch);
    for (int y = 0; y < ch[1].rows; ++y) {
        uchar* pcr = ch[1].ptr<uchar>(y);
        uchar* pcb = ch[2].ptr<uchar>(y);
        const float* g = gSm.ptr<float>(y);
        for (int x = 0; x < ch[1].cols; ++x) {
            pcr[x] = toU8(128.0 + ((double)pcr[x] - 128.0) * (double)g[x]);
            pcb[x] = toU8(128.0 + ((double)pcb[x] - 128.0) * (double)g[x]);
        }
    }
    if (chromaPostSigma_ > 0.01) {
        cv::GaussianBlur(ch[1], ch[1], cv::Size(0, 0), chromaPostSigma_);
        cv::GaussianBlur(ch[2], ch[2], cv::Size(0, 0), chromaPostSigma_);
    }
    cv::merge(ch, ycc);
    cv::Mat out;
    cv::cvtColor(ycc, out, cv::COLOR_YCrCb2BGR);
    return out;
}

void myColorCorrect::_neutralizeCast(cv::Mat& bgr, double strength) const
{
    cv::Mat f;
    bgr.convertTo(f, CV_32FC3);
    std::vector<cv::Mat> ch;
    cv::split(f, ch);
    cv::Mat L = kLumB * ch[0] + kLumG * ch[1] + kLumR * ch[2];
    cv::Mat gb = ch[1] - 0.5 * (ch[0] + ch[2]);

    double sum[3] = {0.0, 0.0, 0.0};
    double cnt[3] = {0.0, 0.0, 0.0};
    for (int y = 0; y < L.rows; ++y) {
        const float* pl = L.ptr<float>(y);
        const float* pg = gb.ptr<float>(y);
        for (int x = 0; x < L.cols; ++x) {
            const double l = pl[x];
            const double v = pg[x];
            if (l < 85.0) { sum[0] += v; cnt[0] += 1.0; }
            else if (l <= 170.0) { sum[1] += v; cnt[1] += 1.0; }
            else { sum[2] += v; cnt[2] += 1.0; }
        }
    }
    double val[3];
    for (int i = 0; i < 3; ++i) val[i] = (cnt[i] > 0.0) ? sum[i] / cnt[i] : 0.0;

    const double node[3] = {60.0, 127.5, 200.0};
    for (int y = 0; y < L.rows; ++y) {
        const float* pl = L.ptr<float>(y);
        cv::Vec3f* pb = f.ptr<cv::Vec3f>(y);
        for (int x = 0; x < L.cols; ++x) {
            const double l = pl[x];
            double corr = val[0];
            if (l > node[2]) {
                corr = val[2];
            } else if (l >= node[0]) {
                if (l <= node[1]) {
                    const double t = (l - node[0]) / (node[1] - node[0]);
                    corr = val[0] * (1.0 - t) + val[1] * t;
                } else {
                    const double t = (l - node[1]) / (node[2] - node[1]);
                    corr = val[1] * (1.0 - t) + val[2] * t;
                }
            }
            pb[x][2] = (float)clamp255((double)pb[x][2] + corr * strength); // R
            pb[x][0] = (float)clamp255((double)pb[x][0] + corr * strength); // B
        }
    }
    f.convertTo(bgr, CV_8U);
}

cv::Mat myColorCorrect::_protect(const cv::Mat& bgr, double satMax, bool warm, bool sharpen) const
{
    cv::Mat v;
    cv::bilateralFilter(bgr, v, bilatD_, bilatSigmaColor_, bilatSigmaSpace_);

    {
        cv::Mat x;
        v.convertTo(x, CV_32F, 1.0 / 255.0);
        for (int y = 0; y < x.rows; ++y) {
            float* q = x.ptr<float>(y);
            for (int c = 0; c < x.cols * 3; ++c) {
                const double t = (double)q[c];
                const double s = t + scStrength_ * (t - 0.5) * (4.0 * t * (1.0 - t));
                q[c] = (float)clampd(s, 0.0, 1.0);
            }
        }
        x.convertTo(v, CV_8U, 255.0);
    }

    for (int y = 0; y < v.rows; ++y) {
        cv::Vec3b* q = v.ptr<cv::Vec3b>(y);
        for (int x = 0; x < v.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                double val = (double)q[x][c];
                if (val > hlKnee_) val = hlKnee_ + (val - hlKnee_) * hlComp_;
                q[x][c] = toU8(val);
            }
        }
    }

    v = _chromaDenoise(v);

    v = _hueSelectiveSaturation(v, satMax);
    v = _chromaDenoise(v);

    _neutralizeCast(v, 1.0);
    _neutralizeCast(v, 1.0);

    if (warm) {
        for (int y = 0; y < v.rows; ++y) {
            cv::Vec3b* q = v.ptr<cv::Vec3b>(y);
            for (int x = 0; x < v.cols; ++x) {
                q[x][2] = toU8((double)q[x][2] * warmRScale_);
                q[x][0] = toU8((double)q[x][0] * warmBScale_);
            }
        }
    }

    if (sharpen) {
        cv::Mat blur;
        cv::GaussianBlur(v, blur, cv::Size(0, 0), unsharpSigma_);
        cv::Mat vf, bf, uf;
        v.convertTo(vf, CV_32F);
        blur.convertTo(bf, CV_32F);
        uf = vf + unsharpAmount_ * (vf - bf);
        uf.convertTo(v, CV_8U);
    }
    return v;
}
namespace {

struct ChStat { double mean; double p1; double p50; double p99; double p999; int mx; long long n255; };

ChStat channelStat(const cv::Mat& bgr, int ch)
{
    long long hist[256];
    for (int i = 0; i < 256; ++i) hist[i] = 0;
    double sum = 0.0;
    const int rows = bgr.rows, cols = bgr.cols;
    for (int y = 0; y < rows; ++y) {
        const cv::Vec3b* p = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; ++x) {
            const int v = p[x][ch];
            hist[v]++;
            sum += (double)v;
        }
    }
    ChStat s;
    const double total = (double)rows * (double)cols;
    s.mean = sum / total;
    s.n255 = hist[255];
    s.mx = 255;
    while (s.mx > 0 && hist[s.mx] == 0) --s.mx;
    const double ps[4] = {1.0, 50.0, 99.0, 99.9};
    double vals[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
        const double want = total * ps[k] / 100.0;
        double acc = 0.0;
        for (int i = 0; i < 256; ++i) {
            acc += (double)hist[i];
            if (acc >= want) { vals[k] = (double)i; break; }
        }
    }
    s.p1 = vals[0]; s.p50 = vals[1]; s.p99 = vals[2]; s.p999 = vals[3];
    return s;
}

} // namespace

double myColorCorrect::_highFreqStd(const cv::Mat& image, int channel) const
{
    cv::Mat f;
    image.convertTo(f, CV_32F);
    std::vector<cv::Mat> chs;
    cv::split(f, chs);
    cv::Mat k = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    cv::Mat r;
    cv::filter2D(chs[channel], r, CV_32F, k);
    cv::Scalar m, sd;
    cv::meanStdDev(r, m, sd);
    return sd[0];
}

void myColorCorrect::_reportMetrics(const cv::Mat& image, const std::string& tag) const
{
    cv::Mat f;
    image.convertTo(f, CV_32F);
    std::vector<cv::Mat> chs;
    cv::split(f, chs);
    cv::Mat L = kLumB * chs[0] + kLumG * chs[1] + kLumR * chs[2];
    cv::Mat gb = chs[1] - 0.5 * (chs[0] + chs[2]);

    cv::Scalar mL, sL;
    cv::meanStdDev(L, mL, sL);

    double cntAll = 0.0, sGlob = 0.0, sDark = 0.0, sMid = 0.0, sBright = 0.0;
    double nDark = 0.0, nMid = 0.0, nBright = 0.0;
    for (int y = 0; y < L.rows; ++y) {
        const float* pl = L.ptr<float>(y);
        const float* pg = gb.ptr<float>(y);
        for (int x = 0; x < L.cols; ++x) {
            const double l = pl[x];
            const double v = pg[x];
            sGlob += v;
            cntAll += 1.0;
            if (l < 85.0) { sDark += v; nDark += 1.0; }
            else if (l <= 170.0) { sMid += v; nMid += 1.0; }
            else { sBright += v; nBright += 1.0; }
        }
    }

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    double sSum = 0.0;
    for (int y = 0; y < hsv.rows; ++y) {
        const cv::Vec3b* p = hsv.ptr<cv::Vec3b>(y);
        for (int x = 0; x < hsv.cols; ++x) sSum += (double)p[x][1];
    }

    long long n255 = 0;
    {
        const int rows = image.rows, cols = image.cols;
        for (int y = 0; y < rows; ++y) {
            const cv::Vec3b* p = image.ptr<cv::Vec3b>(y);
            for (int x = 0; x < cols; ++x) {
                if (p[x][0] == 255 || p[x][1] == 255 || p[x][2] == 255) n255++;
            }
        }
    }

    std::cout << "[" << tag << "] glob=" << cv::format("%+.2f", sGlob / cntAll)
              << " dark=" << cv::format("%+.2f", nDark > 0 ? sDark / nDark : 0.0)
              << "(" << (long long)nDark << ")"
              << " mid=" << cv::format("%+.2f", nMid > 0 ? sMid / nMid : 0.0)
              << "(" << (long long)nMid << ")"
              << " bright=" << cv::format("%+.2f", nBright > 0 ? sBright / nBright : 0.0)
              << "(" << (long long)nBright << ")"
              << " | S=" << cv::format("%.1f", sSum / cntAll)
              << " | std=" << cv::format("%.2f", sL[0])
              << " | n255%=" << cv::format("%.4f", 100.0 * (double)n255 / cntAll)
              << " | hf=[B " << cv::format("%.1f", _highFreqStd(image, 0))
              << ", G " << cv::format("%.1f", _highFreqStd(image, 1))
              << ", R " << cv::format("%.1f", _highFreqStd(image, 2)) << "]" << std::endl;

    const char* names[3] = {"B", "G", "R"};
    for (int c = 0; c < 3; ++c) {
        ChStat st = channelStat(image, c);
        std::cout << "    " << names[c]
                  << " mean=" << cv::format("%6.1f", st.mean)
                  << " p1=" << (int)st.p1 << " p50=" << (int)st.p50
                  << " p99=" << cv::format("%5.1f", st.p99)
                  << " p999=" << cv::format("%5.1f", st.p999)
                  << " max=" << st.mx << " n255=" << st.n255 << std::endl;
    }
}

void myColorCorrect::_reportParams() const
{
    std::cout << "[params] bands=" << bands_
              << " trim=" << cv::format("%.2f", trimRatio_)
              << " smoothWin=" << smoothWin_
              << " sCurve=" << cv::format("%.2f", scStrength_)
              << " hlRolloff=" << cv::format("%.0f/%.1f", hlKnee_, hlComp_)
              << " chromaDenoise=" << chromaMedK_ << "+bil" << chromaBilD_
              << "/" << cv::format("%.0f/%.0f", chromaBilSigmaColor_, chromaBilSigmaSpace_)
              << " hueSel(max=" << cv::format("%.2f", hueSatMax_)
              << ",d>=" << cv::format("%.0f", hueLo_) << "+" << cv::format("%.0f", hueSpan_)
              << ",S>=" << cv::format("%.0f", satLo_) << "+" << cv::format("%.0f", satSpan_)
              << ",gsig=" << cv::format("%.1f", gainSmoothSigma_) << ")"
              << " warm=" << cv::format("%.3f/%.3f", warmRScale_, warmBScale_)
              << " unsharp=" << cv::format("%.2f/sigma%.1f", unsharpAmount_, unsharpSigma_)
              << " bilateral=" << bilatD_ << "/" << cv::format("%.0f/%.0f", bilatSigmaColor_, bilatSigmaSpace_)
              << std::endl;
}
namespace {

cv::Mat montageH(const cv::Mat& a, const cv::Mat& b, int gap)
{
    cv::Mat out(a.rows, a.cols + gap + b.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    a.copyTo(out(cv::Rect(0, 0, a.cols, a.rows)));
    b.copyTo(out(cv::Rect(a.cols + gap, 0, b.cols, b.rows)));
    return out;
}

cv::Mat montageV(const cv::Mat& a, const cv::Mat& b, int gap)
{
    cv::Mat out(a.rows + gap + b.rows, a.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    a.copyTo(out(cv::Rect(0, 0, a.cols, a.rows)));
    b.copyTo(out(cv::Rect(0, a.rows + gap, b.cols, b.rows)));
    return out;
}

cv::Mat fitHeight(const cv::Mat& src, int h)
{
    const double s = (double)h / (double)src.rows;
    cv::Mat r;
    cv::resize(src, r, cv::Size((int)(src.cols * s + 0.5), h), 0, 0, cv::INTER_AREA);
    return r;
}

bool ensureDir(const std::string& path)
{
    if (dirExists(path)) return true;
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            if (cur == "/" || cur.empty()) continue;
            if (!dirExists(cur) && ::mkdir(cur.c_str(), 0755) != 0 && !dirExists(cur)) {
                return false;
            }
        }
    }
    return dirExists(path);
}

} // namespace

bool myColorCorrect::_saveOutputs() const
{
    if (!ensureDir(outputDir_)) {
        std::cerr << "[ERR] cannot create " << outputDir_ << std::endl;
        return false;
    }
    if (!ensureDir(expDir_)) {
        std::cerr << "[ERR] cannot create " << expDir_ << std::endl;
        return false;
    }

    const std::string corrected = outputDir_ + "/corrected.png";
    if (!cv::imwrite(corrected, dst_)) {
        std::cerr << "[ERR] cannot write " << corrected << std::endl;
        return false;
    }
    std::cout << "[OUT] " << corrected << std::endl;

    {
        cv::Mat left = fitHeight(src_, 800);
        cv::Mat right = fitHeight(dst_, 800);
        cv::Mat cmp = montageH(left, right, 10);
        const std::string cp = outputDir_ + "/compare.jpg";
        cv::imwrite(cp, cmp, std::vector<int>(2, 0));
        std::cout << "[OUT] " << cp << std::endl;
    }

    return true;
}