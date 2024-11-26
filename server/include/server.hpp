/**
 * @file server.hpp
 * @brief Handles client connections and responses for video streaming and detection requests.
 * 
 * This file contains functions that handle various client requests for video streaming
 * and vehicle detection.
 * 
 * Supported endpoints:
 * - /start_stream : Start video streaming.
 * - /resume_stream : Resume video streaming.
 * - /pause_stream : Pause video streaming.
 * - /start_detection : Start vehicle detection in the video.
 */
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


/**
 * @class server::rtp::app
 * @brief Handles video streaming and detection requests.
 *
 * This class provides functions to handle various client requests, including starting, pausing,
 * and resuming video streaming, as well as starting vehicle detection.
 */
namespace server {
    namespace rtp {
        /**
         * @class app
         * @brief Handles video streaming and detection requests.
         *
         * This class provides functions to handle various client requests, including starting, pausing,
         * and resuming video streaming, as well as starting vehicle detection.
         */
        struct app {
            uint16_t port_num = 5004;

            app& port(uint16_t p) {
                port_num = p;
                return *this;
            }
            /**
             * @brief Handles client connections and provides responses for each request.
             * 
             * Supported endpoints:
             * - /start_stream : Start video streaming.
             * - /resume_stream : Resume video streaming.
             * - /pause_stream : Pause video streaming.
             * - /start_detection : Start vehicle detection in the video.
             * 
             * @param socket The client socket connection.
             */
            void handle_streaming_request(std::shared_ptr<boost::asio::ip::tcp::socket> shared_socket);
            // GStreamer 파이프라인
            std::string get_gstream_pipeline() const;
            // 비디오 스트리밍 함수
            void start_streaming(const std::string& video_path, std::shared_ptr<boost::asio::ip::tcp::socket> shared_socket);   
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
