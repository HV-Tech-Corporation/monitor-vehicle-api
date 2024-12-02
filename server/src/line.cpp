#include "line.hpp"

namespace api {
    namespace detection { 
        namespace line {
            std::vector<cv::Point> lane1 = {
                cv::Point(147, 0), cv::Point(203, 68), cv::Point(314, 222), 
                cv::Point(453, 510), cv::Point(457, 572), cv::Point(437, 720)
            };

            std::vector<cv::Point> lane2 = {
                cv::Point(424, 0), cv::Point(493, 66), cv::Point(503, 79),
                cv::Point(835, 424), cv::Point(897, 543), cv::Point(907, 720)
            };
        } // line namespace
    } // detection namespace
} // api namespace

void api::detection::line::drawLine(cv::Mat& frame, const std::vector<std::vector<cv::Point>> &lane, const Scalar& color) {
    for (const auto& roi : lane ) {
        cv::fillConvexPoly(frame, roi, color, LINE_AA);
    } 
}

int api::detection::line::checkPosition(const cv::Point& point) {
    // 보간된 lane1과 lane2의 좌표 계산
    int lane1_x = INT_MIN;
    int lane2_x = INT_MAX;

    for (size_t i = 1; i < lane1.size(); ++i) {
        if (point.y >= lane1[i - 1].y && point.y <= lane1[i].y) {
            float t = (float)(point.y - lane1[i - 1].y) / (lane1[i].y - lane1[i - 1].y); // 비율 계산
            lane1_x = lane1[i - 1].x + t * (lane1[i].x - lane1[i - 1].x);
            break;
        }
    }

    for (size_t i = 1; i < lane2.size(); ++i) {
        if (point.y >= lane2[i - 1].y && point.y <= lane2[i].y) {
            float t = (float)(point.y - lane2[i - 1].y) / (lane2[i].y - lane2[i - 1].y); // 비율 계산
            lane2_x = lane2[i - 1].x + t * (lane2[i].x - lane2[i - 1].x);
            break;
        }
    }

    // 점의 위치 판단
    if (point.x < lane1_x) {
        return 1; // lane1 왼쪽
    }
    if (point.x > lane2_x) {
        return 3; // lane2 오른쪽
    }
    if (point.x >= lane1_x && point.x <= lane2_x) {
        return 2; // lane1과 lane2 사이
    }

    return 0; // 기타
}

