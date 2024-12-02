#ifndef LINE_GENERATOR_
#define LINE_GENERATOR_


#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

namespace api {
    namespace detection { 
        namespace line {
            extern std::vector<cv::Point> lane1;
            extern std::vector<cv::Point> lane2;

            void drawLine(cv::Mat& frame, const std::vector<std::vector<cv::Point>> &lane, const Scalar& color);
            int checkPosition(const cv::Point& point);
            
        } //line namespace
    } //detection namespace
} // api namespace

#endif