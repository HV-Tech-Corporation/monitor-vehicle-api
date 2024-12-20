#pragma once

#include <opencv2/core.hpp>
#include "kalman_filter.hpp"

class Track {
public:
    // Constructor
    Track();

    // Destructor
    ~Track() = default;

    void Init(const cv::Rect& bbox);
    void Predict();
    void Update(const cv::Rect& bbox);
    cv::Rect GetStateAsBbox() const;
    std::pair<cv::Point, cv::Point> GetStateAsLine() const;
    cv::Point GetPredictedPoint() const;
    
    float GetNIS() const;

    int coast_cycles_ = 0, hit_streak_ = 0;

private:
    Eigen::VectorXd ConvertBboxToObservation(const cv::Rect& bbox) const;
    cv::Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;
    std::pair<cv::Point, cv::Point> ConvertStateToLine(const Eigen::VectorXd &state) const;
    
    KalmanFilter kf_;
};