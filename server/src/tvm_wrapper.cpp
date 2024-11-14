#include "tvm_wrapper.hpp"

namespace api {
    namespace detection {
        tvm::runtime::Module loaded_lib;
        tvm::runtime::Module mod;
        tvm::runtime::PackedFunc set_input;
        tvm::runtime::PackedFunc run;
        tvm::runtime::PackedFunc get_output;
        DLDevice dev;
    }
}

// DetectedObject structure
struct DetectedObject {
    int id;
    cv::Rect boxes;
    cv::KalmanFilter kalman;
    float confidences;
    int class_ids;
    bool matched = false;
    int lost_frames = 0;  // To handle occlusion and disappearance
};

// IoU(Intersection over Union) 계산 함수
float IoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;

    return static_cast<float>(interArea) / (box1Area + box2Area - interArea);
}


// // Non-Maximum Suppression (NMS) 함수
std::vector<int> NonMaximumSuppression(const std::vector<DetectedObject>& tracking_object_group, float iou_threshold) {
    std::vector<int> indices;
    std::vector<int> sorted_indices(tracking_object_group.size());

    // 각 tracking_object의 cv::Rect 값을 추출하여 boxes 벡터에 추가
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (const auto& obj : tracking_object_group) {
        boxes.push_back(obj.boxes);  // 올바른 멤버 접근
        confidences.push_back(obj.confidences);
    }

    // 신뢰도를 기준으로 내림차순 정렬
    for (int i = 0; i < boxes.size(); ++i) {
        sorted_indices[i] = i;
    }

    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int i, int j) {
        return confidences[i] > confidences[j];
    });

    std::vector<bool> suppressed(boxes.size(), false);
    for (int i = 0; i < sorted_indices.size(); ++i) {
        int idx = sorted_indices[i];
        if (suppressed[idx]) continue;

        indices.push_back(idx);

        for (int j = i + 1; j < sorted_indices.size(); ++j) {
            int next_idx = sorted_indices[j];
            if (suppressed[next_idx]) continue;

            // IoU 계산 후 겹치면 suppress
            if (IoU(boxes[idx], boxes[next_idx]) > iou_threshold) {
                suppressed[next_idx] = true;
            }
        }
    }
    return indices;
}

tvm::runtime::NDArray preprocess_frame(cv::Mat& frame,int _batch, int _input_w, int _input_h) {
    cv::Mat resized_img;
    cv::resize(frame, resized_img, cv::Size(640, 640));
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);

    int batch = _batch;
    int height = _input_h;
    int width = _input_w;
    int channels = 3;

    std::vector<int64_t> shape = {batch, channels, height, width};
    tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty(shape, DLDataType{kDLFloat, 32, 1}, DLDevice{kDLCPU, 0});

    // OpenCV의 Mat는 HWC 형식 (height, width, channels)이고 NDArray는 NCHW 형식이므로 데이터를 변환하면서 바로 복사하도록 했습니다.
    float* input_data = static_cast<float*>(input->data);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv::Vec3f pixel = resized_img.at<cv::Vec3f>(h, w);
            input_data[0 * height * width + h * width + w] = pixel[0];  // R 채널
            input_data[1 * height * width + h * width + w] = pixel[1];  // G 채널
            input_data[2 * height * width + h * width + w] = pixel[2];  // B 채널
        }
    }

    return input;  // tvm::runtime::NDArray 반환
}

void api::detection::load_model(const std::string& model_path, const std::string device) {
    loaded_lib = tvm::runtime::Module::LoadFromFile(model_path);
    if(device == "gpu") dev = {kDLCUDA, 0};
    else if (device == "cpu") dev = {kDLCPU, 0};
    mod = loaded_lib.GetFunction("default")(dev);
    set_input = mod.GetFunction("set_input");
    run = mod.GetFunction("run");
    get_output = mod.GetFunction("get_output");
}

std::vector<cv::Rect> ProcessYOLOOutput(tvm::runtime::NDArray output, const std::vector<std::string>& class_names, cv::Mat& frame, 
                       std::vector<DetectedObject>& trackedObjects, int& currentID, float conf_threshold = 0.75) {

    // LOG(INFO) << "detection start...";

    const int64_t* shape = output.Shape().data();
    int num_detections = shape[1];  // 감지된 객체 수

    float* output_data = static_cast<float*>(output->data);
    int num_classes = class_names.size();  // 클래스 개수

    int original_width = frame.cols;
    int original_height = frame.rows;
    const float width_ratio = static_cast<float>(original_width) / 640.0f;
    const float height_ratio = static_cast<float>(original_height) / 640.0f;

    const int data_stride = 5 + num_classes;

    std::vector<DetectedObject> detected_objects;
    
    // Extract detected objects
    for (int i = 0; i < num_detections; ++i) {
        int base_index = i * data_stride;

        float cx = output_data[base_index + 0] * width_ratio;
        float cy = output_data[base_index + 1] * height_ratio;
        float w = output_data[base_index + 2] * width_ratio;
        float h = output_data[base_index + 3] * height_ratio;
        float confidence = output_data[base_index + 4];

        if (confidence > conf_threshold) {
            float* class_scores = &output_data[base_index + 5];
            int class_id = std::distance(class_scores, std::max_element(class_scores, class_scores + num_classes));

            if (class_id == 1 || class_id == 2 || class_id ==3  || class_id == 5) {  // Only process "person" class
                int x1 = static_cast<int>(cx - (w / 2));
                int y1 = static_cast<int>(cy - (h / 2));
                int x2 = static_cast<int>(cx + (w / 2));
                int y2 = static_cast<int>(cy + (h / 2));

                DetectedObject obj;
                obj.id = -1;  // Will be assigned later
                obj.boxes = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                obj.confidences = confidence;
                obj.class_ids = class_id;
                detected_objects.push_back(obj);
                
            }
        }

    }
    float iou_threshold = 0.4f; 
    std::vector<int> nms_indices = NonMaximumSuppression(detected_objects, iou_threshold);
    std::vector<cv::Rect> bbox_per_frame;

    // Display the tracked objects
    for (int idx : nms_indices) {
        bbox_per_frame.push_back(detected_objects[idx].boxes);
        std::string label = class_names[detected_objects[idx].class_ids];
    }

    return bbox_per_frame;
}

// detection.cpp
void api::detection::detect(
    std::atomic<bool>& start_detection,
    std::atomic<bool>& pause_detection,
    std::mutex& frame_mutex,
    std::condition_variable& detection_cv,
    cv::Mat& shared_frame
) {
    LOG(INFO) << "Detection start";
    while (start_detection) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        detection_cv.wait(lock, [&pause_detection, &start_detection] { 
            return !pause_detection || !start_detection; 
        });

        if (!start_detection) {
            break;
        }

        if (shared_frame.empty()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        cv::Mat frame = shared_frame.clone();
        lock.unlock();
        tvm::runtime::NDArray input =  preprocess_frame(frame, 1, 640, 640);
        set_input("input", input);
        run();
        tvm::runtime::NDArray output = get_output(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        std::cout << "test" << std::endl;
    }
    LOG(INFO) << "Detection Complete";
}
