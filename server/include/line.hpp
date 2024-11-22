#ifndef LINE_GENERATOR_
#define LINE_GENERATOR_


#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

namespace api {
    namespace detection { 
        namespace line {
            std::vector<std::vector<cv::Point>> lane1 = {
                { Point(147, 5), Point(162, 3), Point(217, 68), Point(203, 69) },
                { Point(203, 69), Point(217, 68), Point(328, 222), Point(277, 142) },
                { Point(277, 142), Point(328, 222), Point(366, 270), Point(350, 271) },
                { Point(350, 271), Point(366, 270), Point(429, 366), Point(412, 370) },
                { Point(412, 370), Point(429, 366), Point(448, 410), Point(439, 415) },
                { Point(439, 415), Point(448, 410), Point(467, 511), Point(453, 506) },
                { Point(453, 506), Point(467, 511), Point(471, 572), Point(457, 568) },
                { Point(457, 568), Point(471, 572), Point(454, 716), Point(437, 714) }
            };

            std::vector<std::vector<cv::Point>> lane2 = {
                { Point(147, 5), Point(162, 3), Point(217, 68), Point(203, 69) },
                { Point(203, 69), Point(217, 68), Point(328, 222), Point(277, 142) },
                { Point(277, 142), Point(328, 222), Point(366, 270), Point(350, 271) },
                { Point(350, 271), Point(366, 270), Point(429, 366), Point(412, 370) },
                { Point(412, 370), Point(429, 366), Point(448, 410), Point(439, 415) },
                { Point(439, 415), Point(448, 410), Point(467, 511), Point(453, 506) },
                { Point(453, 506), Point(467, 511), Point(471, 572), Point(457, 568) },
                { Point(457, 568), Point(471, 572), Point(454, 716), Point(437, 714) }
            };

            void drawLine(cv::Mat& frame, const std::vector<std::vector<cv::Point>> &lane, const Scalar& color);
            int checkPosition(const cv::Point& point);
            
        } //line namespace
    } //detection namespace
} // api namespace

#endif