#ifndef TVM_WRAPPER_
#define TVM_WRAPPER_

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>
#include <chrono>

namespace api {
    namespace detection {
        const static std::vector<std::string> COCO_CLASS_80 = {
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", 
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", 
            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", 
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", 
            "teddy bear", "hair drier", "toothbrush"
        };

        void load_model(const std::string& model_path, const std::string device = "cpu");

        void detect(
            std::atomic<bool>& start_detection,
            std::atomic<bool>& pause_detection,
            std::atomic<bool>& shared_frame_updated,
            std::mutex& frame_mutex,
            std::condition_variable& detection_cv,
            cv::Mat& shared_frame
        );

        void preprocess_detect(
            const std::string& video_path,
            std::unordered_map<int, cv::Mat>& detected_frames,
            std::mutex& detected_frames_mutex,
            std::unordered_map<int, int>& kruskal_results_per_frames,
            std::mutex& kruskal_results_per_frames_mutex,
            std::atomic<bool>& preload_complete,
            std::mutex& bestShot_mutex,
            cv::Mat bestShot_frame
        );

        extern tvm::runtime::Module loaded_lib;
        extern tvm::runtime::Module mod;
        extern tvm::runtime::PackedFunc set_input;
        extern tvm::runtime::PackedFunc run;
        extern tvm::runtime::PackedFunc get_output;
        extern DLDevice dev;
    } // namespace detection
}// namsespace api
#endif // TVM_WRAPPER_