#include "line.hpp"

void api::detection::line::drawLine(cv::Mat& frame, const std::vector<std::vector<cv::Point>> &lane, const Scalar& color) {
    for (const auto& roi : lane ) {
        cv::fillConvexPoly(frame, roi, color, LINE_AA);
    } 
}

int api::detection::line::checkPosition(const cv::Point& point) {
    bool inLane1 = false, inLane2 = false;

    for(const auto& roi : lane1) {
        if(!roi.empty() && pointPolygonTest(roi, point, false) >= 0) {
            inLane1 = true;
            break;
        }
    }

    for (const auto& roi : lane2) {
        if(!roi.empty() && pointPolygonTest(roi, point, false) >= 0) {
            inLane2 = true;
            break;
        }
    }

    if (inLane1) {
        return 1;
    }
    if (inLane2) {
        return 2;
    }

    for (const auto& roi : lane1) {
        if (!roi.empty()) {
            int minX = roi[0].x;
            for(const auto& pt : roi) {
                if(pt.x < minX) minX = pt.x;
            }
            if (point.x < minX) {
                return 1;
            }
        }
    }

    size_t numPolygons = std::min(lane1.size(), lane2.size());
    for (size_t i = 0; i < numPolygons; ++i) {
        // lane1[i]와 lane2[i]가 충분한 요소를 가지고 있는지 확인
        if (lane1[i].size() > 2 && !lane2[i].empty()) {
            int lane1_right = lane1[i][2].x;  // lane1의 오른쪽 경계 x 좌표
            int lane2_left = lane2[i][0].x;   // lane2의 왼쪽 경계 x 좌표

            if (point.x >= lane1_right && point.x <= lane2_left) {
                return 2;  // lane1과 lane2 사이에 위치
            }
        }
    }

    return 0;  // lane1과 lane2 밖에 위치
}
