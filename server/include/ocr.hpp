#ifndef OCR_WRAPPER_
#define OCR_WRAPPER_

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/ndarray.h>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>
#include <chrono>

namespace api {
    namespace recognition {
        namespace ocr {

            cv::Mat localization(cv::Mat& bbox, const std::string& xml_path) {
                // Haar Cascade 로드
                cv::CascadeClassifier plateCascade;
                if (!plateCascade.load(xml_path)) {
                    std::cerr << "Error loading Haar Cascade file for license plate" << std::endl;
                    return cv::Mat(); // 빈 Mat 반환
                }

                // bbox 영역을 사용하여 bestShot_frame 생성
                cv::Mat bestShot_frame = bbox.clone(); // bbox를 복사
                if (bestShot_frame.empty()) {
                    std::cerr << "Error: Input frame (bbox) is empty" << std::endl;
                    return cv::Mat(); // 빈 Mat 반환
                }

                // plate 탐지
                std::vector<cv::Rect> plates;
                cv::Mat gray;
                cv::cvtColor(bestShot_frame, gray, cv::COLOR_BGR2GRAY); // 그레이스케일 변환
                plateCascade.detectMultiScale(gray, plates, 1.05, 3, 0, cv::Size(20, 20));

                // plates가 비어 있을 경우 처리
                if (plates.empty()) {
                    std::cerr << "No license plates detected" << std::endl;
                    return cv::Mat(); // 빈 Mat 반환
                }

                // plates가 비어 있지 않을 경우 첫 번째 plate 반환
                cv::Mat plate_img = bestShot_frame(plates[0]).clone();
                return plate_img;
            }

            void reconginze(const std::string& model_path, const cv::Mat& image) {
                tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile("/Users/gyujinkim/Desktop/Ai/TVM_TUTORIAL/front/tvm_recognition_fixed.so");
                tvm::runtime::PackedFunc get_func = lib.GetFunction("default");
                DLDevice dev = {kDLCPU, 0};
                tvm::runtime::Module gmod = get_func(dev);

                tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
                tvm::runtime::PackedFunc run = gmod.GetFunction("run");
                tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");

                // 이미지 전처리 및 입력 설정
                tvm::runtime::NDArray input = preprocess_image(image, 100, 48);
                set_input("x", input);

                // 모델 실행
                run();

                // 모델 출력 추출
                tvm::runtime::NDArray output = get_output(0);
                float* output_data = static_cast<float*>(output->data);
                int T = output.Shape()[1];
                int C = output.Shape()[2];

                 std::vector<int> char_indices;
                for (int t = 0; t < T; ++t) {
                    float* start = output_data + t * C;
                    float* end = start + C;
                    int max_index = std::distance(start, std::max_element(start, end));
                    char_indices.push_back(max_index);
                }

                // CTC 디코딩을 통해 텍스트 생성
                std::string result_text = decode_ctc(char_indices);
                float confidence = *std::max_element(output_data, output_data + T * C);

                return {result_text, confidence};
            } // namespace ocr
        } // namespcae recognimtion
    }
}
