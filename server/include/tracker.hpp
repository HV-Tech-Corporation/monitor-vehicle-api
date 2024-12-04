#pragma once

#include <map>
#include <opencv2/core.hpp>

#include "track.hpp"
#include "munkres.hpp"
#include "utils.hpp"

class ObjectTracker {
public:
    ObjectTracker();
    ~ObjectTracker() = default;

    static float CalculateIou(const cv::Rect& det, const Track& track);

    static void HungarianMatching(const std::vector<std::vector<float>>& iou_matrix,
                           size_t nrows, size_t ncols,
                           std::vector<std::vector<float>>& association);

/**
 * Assigns detections to tracked object (both represented as bounding boxes)
 * Returns 2 lists of matches, unmatched_detections
 * @param detection
 * @param tracks
 * @param matched
 * @param unmatched_det
 * @param iou_threshold
 */
    static void AssociateDetectionsToTrackers(const std::vector<std::pair<int, cv::Rect>>& detection,
                                       std::map<int, std::pair<int, Track>>& tracks,
                                       std::map<int, std::pair<int, cv::Rect>>& matched,
                                       std::vector<std::pair<int, cv::Rect>>& unmatched_det,
                                       float iou_threshold = 0.3);

    void Run(const std::vector<std::pair<int, cv::Rect>>& detections);

    std::map<int, std::pair<int, Track>> GetTracks();

private:
    // Hash-map between ID and corresponding tracker
    std::map<int, std::pair<int, Track>> tracks_;

    // Assigned ID for each bounding box
    int id_;
};