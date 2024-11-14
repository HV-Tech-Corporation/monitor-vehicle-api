#ifndef SERVER_HPP
#define SERVER_HPP

#include <boost/asio.hpp>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "response.hpp" // 응답 관리를 위한 response.hpp
#include "tvm_wrapper.hpp"

namespace server {
    namespace rtp {
        struct app {
            uint16_t port_num = 5004;

            app& port(uint16_t p) {
                port_num = p;
                return *this;
            }
            // 클라이언트 요청을 처리하는 함수
            void handle_streaming_request(boost::asio::ip::tcp::socket& socket);
            // GStreamer 파이프라인
            std::string get_gstream_pipeline() const;
            // 비디오 스트리밍 함수
            void start_streaming(const std::string& video_path);   
        }; //struct app

        
        // 서버 시작 함수
        void start_server(uint16_t port);  

        const std::string video_path = "/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/traffic_jam2.mp4";
        extern std::atomic<bool> pause_streaming;
        extern std::atomic<bool> rewind_streaming;
        extern std::atomic<bool> start_detection;
        extern std::atomic<bool> pause_detection;

        extern std::condition_variable detection_cv;
        extern std::mutex frame_mutex;
        extern cv::Mat shared_frame;

        
    } // namespace rtp
} // namespace server

#endif // SERVER_HPP
