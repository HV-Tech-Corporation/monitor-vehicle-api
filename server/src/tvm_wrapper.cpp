#include "tvm_wrapper.hpp"
#include "tracker.hpp"
#include "line.hpp"

namespace api {
    namespace detection {
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

        tvm::runtime::Module loaded_lib;
        tvm::runtime::Module mod;
        tvm::runtime::PackedFunc set_input;
        tvm::runtime::PackedFunc run;
        tvm::runtime::PackedFunc get_output;
        DLDevice dev;
        std::set<int> saved_ids;
        std::vector<DetectedObject> trackedObjects;
        int currentId = 0;

        ObjectTracker tracker;
    }
}

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
std::vector<int> NonMaximumSuppression(const std::vector<api::detection::DetectedObject>& tracking_object_group, float iou_threshold) {
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

std::vector<api::detection::DetectedObject> ProcessYOLOOutput(tvm::runtime::NDArray output, const std::vector<std::string>& class_names, cv::Mat& frame, 
                       std::vector<api::detection::DetectedObject>& trackedObjects, int& currentID, float conf_threshold = 0.8) {

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

    std::vector<api::detection::DetectedObject> detected_objects;
    
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

            if (class_id == 1 || class_id == 2 || class_id ==3  || class_id == 5) {  
                int x1 = static_cast<int>(cx - (w / 2));
                int y1 = static_cast<int>(cy - (h / 2));
                int x2 = static_cast<int>(cx + (w / 2));
                int y2 = static_cast<int>(cy + (h / 2));

                api::detection::DetectedObject obj;
                obj.id = -1;  // Will be assigned later
                obj.boxes = cv::Rect(x1, y1, x2 - x1, y2 - y1);
                obj.confidences = confidence;
                obj.class_ids = class_id;
                detected_objects.push_back(obj);
                
            }
        }

    }
    float iou_threshold = 0.55f; 
    std::vector<int> nms_indices = NonMaximumSuppression(detected_objects, iou_threshold);
    std::vector<cv::Rect> bbox_per_frame;

    // Display the tracked objects
    for (int idx : nms_indices) {
        bbox_per_frame.push_back(detected_objects[idx].boxes);
        std::string label = class_names[detected_objects[idx].class_ids];
    }

    return detected_objects;
}

// detection.cpp
void api::detection::detect(
    std::atomic<bool>& start_detection,
    std::atomic<bool>& pause_detection,
    std::atomic<bool>& shared_frame_updated,
    std::mutex& frame_mutex,
    std::condition_variable& detection_cv,
    cv::Mat& shared_frame
) {
    LOG(INFO) << "Detection start";
    
    while (start_detection) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        detection_cv.wait(lock, [&pause_detection, &start_detection] { 
            return !pause_detection || start_detection; 
        });

        if (!start_detection) {
            break;
        }


        if (shared_frame.empty()) {
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        // Detection과 트래킹을 shared_frame에서 직접 수행
        tvm::runtime::NDArray input = preprocess_frame(shared_frame, 1, 640, 640);
        set_input("input", input);
        run();
        tvm::runtime::NDArray output = get_output(0);

        // YOLO 및 트래킹 결과 계산
        std::vector<api::detection::DetectedObject> yolo_output = ProcessYOLOOutput(output, COCO_CLASS_80, shared_frame, trackedObjects, currentId, 0.45);
        float iou_threshold = 0.4f;
        std::vector<int> nms_indices = NonMaximumSuppression(yolo_output, iou_threshold);
        std::vector<cv::Rect> all_detections;

        for (int idx : nms_indices) {
            all_detections.push_back(yolo_output[idx].boxes);
        }

        tracker.Run(all_detections);
        const auto tracks = tracker.GetTracks();

        // detection 및 트래킹 결과를 shared_frame에 업데이트
        for (const auto& det : all_detections) {
            cv::rectangle(shared_frame, det, cv::Scalar(0, 0, 255), 2);
        }

        for (auto& trk : tracks) {
            if (trk.second.coast_cycles_ < kMaxCoastCycles && (trk.second.hit_streak_ >= kMinHits)) {
                const auto& bbox = trk.second.GetStateAsBbox();
                if (saved_ids.find(trk.first) == saved_ids.end()) {
                    if (bbox.x >= 0 && bbox.y >= 0 && bbox.x + bbox.width <= shared_frame.cols 
                    && bbox.y + bbox.height <= shared_frame.rows) {
                        // 아직 저장되지 않은 새로운 ID이므로 바운더리 박스를 이미지로 저장
                        cv::Mat cropped_image = shared_frame(bbox).clone(); // 바운더리 박스 영역만 잘라내기
                        // cv::rectangle(shared_frame, bbox, cv::Scalar(0, 0, 255), 2);
                        std::string filename = "object_" + std::to_string(trk.first) + ".png";
                        cv::imwrite(filename, cropped_image);
                        // ID를 saved_ids에 추가하여 이후에는 저장되지 않도록 함
                        saved_ids.insert(trk.first);
                    }
                }
            }
        }
        
        shared_frame_updated.store(true);
        detection_cv.notify_one();
        
        lock.unlock();  // 락을 해제하여 외부에서 frame을 접근할 수 있도록 함
        std::this_thread::sleep_for(std::chrono::milliseconds(30));  // 대기 시간 조정
    }
    LOG(INFO) << "Detection Complete";
}


void api::detection::preprocess_detect(
    const std::string& video_path,
    std::unordered_map<int, cv::Mat>& detected_frames,
    std::mutex& detected_frames_mutex,
    std::atomic<bool>& preload_complete,
    std::mutex& bestShot_mutex,
    cv::Mat bestShot_frame
) {
    LOG(INFO) << "Preloading and Detection Start";
    
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file for detection" << std::endl;
        return;
    }

    std::unordered_set<int> processed_ids;
    std::map<int, std::vector<cv::Point>> points_by_bbox;
    std::map<int, int> vehicle_count_by_bbox;

    int frame_index = 0;  // 현재 프레임 인덱스
    cv::Mat frame;

    ObjectTracker tracker;

    std::map<int, cv::Scalar> color_map = {
        {0, cv::Scalar(0, 255, 0)},  // 초록색
        {1, cv::Scalar(255, 0, 0)},  // 파란색
        {2, cv::Scalar(0, 0, 255)},  // 빨간색
        {3, cv::Scalar(255, 255, 0)} // 노란색
    };

    cv::Scalar default_color(255, 255, 255); 
    

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;  // 비디오 끝에 도달
        }
        
        // YOLO 모델을 사용해 탐지 수행
        tvm::runtime::NDArray input = preprocess_frame(frame, 1, 640, 640);
        set_input("input", input);
        run();
        tvm::runtime::NDArray output = get_output(0);

        points_by_bbox.clear();

        // YOLO 및 트래킹 결과 계산
        std::vector<api::detection::DetectedObject> yolo_output = ProcessYOLOOutput(output, COCO_CLASS_80, frame, trackedObjects, currentId, 0.45);
        float iou_threshold = 0.4f;
        std::vector<int> nms_indices = NonMaximumSuppression(yolo_output, iou_threshold);
        std::vector<std::pair<cv::Rect, int>> all_detections_with_class_ids;

        // 탐지 결과를 복사하고 바운딩 박스를 그립니다.
        cv::Mat detected_frame = frame.clone();
        
        

        for (int idx : nms_indices) {
            const auto& detected_object = yolo_output[idx];
            all_detections_with_class_ids.emplace_back(detected_object.boxes, detected_object.class_ids);
            // cv::putText(detected_frame, class_text,
            //         cv::Point(yolo_output[idx].boxes.tl().x + yolo_output[idx].boxes.width, yolo_output[idx].boxes.y - 10),
            //         cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2); // 흰색 텍스트
        }

        tracker.Run(all_detections);
        const auto tracks = tracker.GetTracks();
        
        // std::cout << "------------------------" << std::endl;
        // std::cout << "nms size is :" << nms_indices.size() << std::endl; 
        // std::cout << "track size is : " << tracks.size() << std::endl;

        vehicle_count_by_bbox.clear();

        // detection 및 트래킹 결과를 shared_frame에 업데이트
        int track_cnt = 0;
        
        // std::cout << "track size  is  : " <<  tracks.size() << std::endl;
        // api::detection::line::drawLine(detected_frame, api::detection::line::lane1, Scalar(255, 0, 0));
        // api::detection::line::drawLine(detected_frame, api::detection::line::lane2, Scalar(0, 255, 0));

        // for(auto &trk : tracks) {
        //     const auto &bbox = trk.second.GetStateAsBbox();
        //     if(trk.second.coast_cycles_ < kMaxCoastCycles && (trk.second.hit_streak_ >= kMinHits || frame_index < kMinHits)) {
        //         std::cout << frame_index << "," << trk.first << "," << bbox.tl().x << "," << bbox.tl().y
        //         << "," << bbox.width << "," << bbox.height << ",1 ,-1, -1, -1"
        //         << " Hit Streak = " << trk.second.hit_streak_ << " Coast Cycles = " << trk.second.coast_cycles_ << std::endl; 
        //     }
        // }

        for (auto &trk : tracks) {
            if (trk.second.coast_cycles_ < kMaxCoastCycles &&
                (trk.second.hit_streak_ >= kMinHits)) {
                const auto &bbox = trk.second.GetStateAsBbox();
                if (bbox.x >= 0 && bbox.y >= 0 && bbox.x + bbox.width <= frame.cols && bbox.y + bbox.height <= frame.rows) {
                    cv::Point predicted_point = trk.second.GetPredictedPoint();
                    cv::Point pos(predicted_point.x, predicted_point.y);
                    cv::circle(detected_frame, pos, 5, cv::Scalar(0, 0, 0), 2);
            
                    int pos_bbox = api::detection::line::checkPosition(pos);
                    vehicle_count_by_bbox[pos_bbox]++;

                    cv::putText(detected_frame, std::to_string(trk.first), cv::Point(bbox.tl().x, bbox.tl().y - 10),
                                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
                    // cv::putText(detected_frame, std::to_string(pos_bbox), cv::Point(bbox.tl().x + bbox.width, bbox.tl().y - 10),
                    //             cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
                    

                    cv::rectangle(detected_frame, bbox, cv::Scalar(0, 0, 255), 2);
                    points_by_bbox[pos_bbox].emplace_back(pos);

                    if (processed_ids.find(trk.first) == processed_ids.end()) {
                        // 새로운 ID일 경우 이미지를 저장
                        cv::Mat bestShot_frame = frame(bbox).clone();
                        cv::imwrite("bestShot_" + std::to_string(frame_index) + "_" + std::to_string(trk.first) +  "_class_ids_" + std::to_string(yolo_output[nms_indices[track_cnt]].class_ids) + ".jpg", bestShot_frame);
                        // 처리된 ID 목록에 추가
                        processed_ids.insert(trk.first);

                        // bestShot frame 업데이트
                        {
                            std::lock_guard<std::mutex> lock(bestShot_mutex);
                            bestShot_frame = frame(bbox).clone();
                        }
                    }

                    track_cnt++;
                }
                
            }   
        }

        


        for (const auto &[bbox_id, count] : vehicle_count_by_bbox) {
            std::string text = "LINE ID: " + std::to_string(bbox_id) + ", CAR: " + std::to_string(count);
            cv::putText(detected_frame, text, cv::Point(50, 50 + bbox_id * 30), // bbox_id에 따라 Y 위치 조정
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }
        
        // 탐지된 결과를 저장
        {
            std::lock_guard<std::mutex> lock(detected_frames_mutex);
            detected_frames[frame_index] = detected_frame;  // 프레임 인덱스에 결과 저장
        }

        for (auto &[id, points] : points_by_bbox) {
            // 노드가 하나라면 그리지 않음
            if (points.size() <= 1) {
                continue;
            }

            // Y축 기준으로 정렬
            std::sort(points.begin(), points.end(), [](const cv::Point &a, const cv::Point &b) {
                return a.y < b.y; // y 좌표 기준 오름차순 정렬
            });

            for (size_t i = 1; i < points.size(); ++i) {
                // 선 길이 계산
                double distance = cv::norm(points[i] - points[i - 1]);
                double max_distance = 500.0; // 예상되는 최대 거리

                // 거리 정규화
                double normalized_distance = std::min(distance / max_distance, 1.0); // 0 ~ 1로 정규화

                // 거리 구간 설정 (세 가지 색상)
                cv::Scalar color; // 색상 초기화
                if (distance < max_distance / 3.0) { // 짧은 거리 (1/3 이하)
                    color = cv::Scalar(50, 50, 255); // 밝은 빨간색
                } else if (distance < 2 * max_distance / 3.0) { // 중간 거리 (1/3 ~ 2/3)
                    color = cv::Scalar(0, 165, 255); // 주황색
                } else { // 긴 거리 (2/3 이상)
                    color = cv::Scalar(0x4A, 0xB2, 0x2C); 
                }

                // 선 그리기
                cv::line(detected_frame, points[i - 1], points[i], color, 2);
            }


            // 오래된 점 삭제 (필요시)
            if (points.size() > 10) { // 최대 10개의 점만 유지
                points.erase(points.begin(), points.end() - 10);
            }
        }


        frame_index++;  // 다음 프레임으로 이동
    }

    preload_complete.store(true);  // 탐지 작업 완료 플래그 설정
    cap.release();
    std::cout << "preprocess done!!" << std::endl;
    LOG(INFO) << "Detection and Preloading Complete";
}
