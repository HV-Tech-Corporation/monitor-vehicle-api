#include "track.hpp"


Track::Track() : kf_(8, 4) {

    /*** Define constant velocity model ***/
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    kf_.F_ <<
           1, 0, 0, 0, 1, 0, 0, 0,
            0, 1, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 1;

    // Give high uncertainty to the unobservable initial velocities
    kf_.P_ <<
           10, 0, 0, 0, 0, 0, 0, 0,
            0, 10, 0, 0, 0, 0, 0, 0,
            0, 0, 10, 0, 0, 0, 0, 0,
            0, 0, 0, 10, 0, 0, 0, 0,
            0, 0, 0, 0, 10000, 0, 0, 0,
            0, 0, 0, 0, 0, 10000, 0, 0,
            0, 0, 0, 0, 0, 0, 10000, 0,
            0, 0, 0, 0, 0, 0, 0, 10000;


    kf_.H_ <<
           1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0;

    kf_.Q_ <<
           1, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0.01, 0, 0, 0,
            0, 0, 0, 0, 0, 0.01, 0, 0,
            0, 0, 0, 0, 0, 0, 0.0001, 0,
            0, 0, 0, 0, 0, 0, 0, 0.0001;

    kf_.R_ <<
           1, 0, 0,  0,
            0, 1, 0,  0,
            0, 0, 10, 0,
            0, 0, 0,  10;
}


// Get predicted locations from existing trackers
// dt is time elapsed between the current and previous measurements
void Track::Predict() {
    kf_.Predict();

    // hit streak count will be reset
    if (coast_cycles_ > 0) {
        hit_streak_ = 0;
    }
    // accumulate coast cycle count
    coast_cycles_++;
}

// 예측된 점을 반환하는 함수 추가
cv::Point Track::GetPredictedPoint() const {
    // Kalman 필터의 예측된 중심점 반환
    return cv::Point(static_cast<int>(kf_.x_predict_(0)), static_cast<int>(kf_.x_predict_(1)));
}


// Update matched trackers with assigned detections
void Track::Update(const cv::Rect& bbox) {

    // get measurement update, reset coast cycle count
    coast_cycles_ = 0;
    // accumulate hit streak count
    hit_streak_++;

    // observation - center_x, center_y, area, ratio
    Eigen::VectorXd observation = ConvertBboxToObservation(bbox);
    kf_.Update(observation);
}

// Create and initialize new trackers for unmatched detections, with initial bounding box
void Track::Init(const cv::Rect &bbox) {
    kf_.x_.head(4) << ConvertBboxToObservation(bbox);
    hit_streak_++;
}


/**
 * Returns the current bounding box estimate
 * @return
 */
cv::Rect Track::GetStateAsBbox() const {
    return ConvertStateToBbox(kf_.x_);
}

std::pair<cv::Point, cv::Point> Track::GetStateAsLine() const {
    return ConvertStateToLine(kf_.x_);
}

float Track::GetNIS() const {
    return kf_.NIS_;
}


/**
 * Takes a bounding box in the form [x, y, width, height] and returns z in the form
 * [x, y, s, r] where x,y is the centre of the box and s is the scale/area and r is
 * the aspect ratio
 *
 * @param bbox
 * @return
 */
Eigen::VectorXd Track::ConvertBboxToObservation(const cv::Rect& bbox) const{
    Eigen::VectorXd observation = Eigen::VectorXd::Zero(4);
    auto width = static_cast<float>(bbox.width);
    auto height = static_cast<float>(bbox.height);
    float center_x = bbox.x + width / 2;
    float center_y = bbox.y + height / 2;
    observation << center_x, center_y, width, height;
    return observation;
}


/**
 * Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
 * [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
 *
 * @param state
 * @return
 */
cv::Rect Track::ConvertStateToBbox(const Eigen::VectorXd &state) const {
    // state - center_x, center_y, width, height, v_cx, v_cy, v_width, v_height
    auto width = std::max(0, static_cast<int>(state[2]));
    auto height = std::max(0, static_cast<int>(state[3]));
    auto tl_x = static_cast<int>(state[0] - width / 2.0);
    auto tl_y = static_cast<int>(state[1] - height / 2.0);
    cv::Rect rect(cv::Point(tl_x, tl_y), cv::Size(width, height));
    return rect;
}

std::pair<cv::Point, cv::Point> Track::ConvertStateToLine(const Eigen::VectorXd &state) const {
    int current_center_x = static_cast<int>(state[0]);
    int current_center_y = static_cast<int>(state[1]);

    int velocity_x = static_cast<int>(state[4] * 10);
    int velocity_y = static_cast<int>(state[5] * 10);

    int predicted_center_x = static_cast<int>(state[0] + velocity_x);
    int predicted_center_y = static_cast<int>(state[1] + velocity_y);

    cv::Point current_position(current_center_x, current_center_y);
    cv::Point predicted_position(predicted_center_x, predicted_center_y);

    return std::make_pair(current_position, predicted_position);
}


