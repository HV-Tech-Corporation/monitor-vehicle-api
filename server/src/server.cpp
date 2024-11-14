#include "server.hpp"

namespace server {
    namespace rtp {
        std::atomic<bool> pause_streaming(false);  
        std::atomic<bool> rewind_streaming(false); 
        std::atomic<bool> start_detection(false);
        std::atomic<bool> pause_detection(false);

        std::condition_variable detection_cv;
        std::mutex frame_mutex;
        cv::Mat shared_frame;

        tvm::runtime::Module loaded_lib;
        tvm::runtime::Module mod;
        tvm::runtime::PackedFunc set_input;
        tvm::runtime::PackedFunc run;
        tvm::runtime::PackedFunc get_output;
        DLDevice dev;
    }
} 

void send_response(boost::asio::ip::tcp::socket& socket, server::http_response::response_type status) {
    server::http_response::response res_inst;
    server::http_response::response _http_response;
    res_inst.set_status(status);
    res_inst.add_header("Content-Type", "text/html");

    // 응답 본문을 설정 (response_type에 따라 본문 선택)
    res_inst.content = _http_response.response_body_to_string(status);

    auto buffers = res_inst.to_buffers();
    boost::asio::write(socket, buffers);
}

void server::rtp::app::handle_streaming_request(boost::asio::ip::tcp::socket& socket) {
    
    try {
        boost::asio::streambuf request;
        boost::asio::read_until(socket, request, "\r\n");

        std::istream request_stream(&request);
        std::string method, path;
        request_stream >> method >> path;

        
        if(method == "GET" && path == "/start_stream") {
            std::cout << "start stream" << std::endl;
            send_response(socket, server::http_response::response_type::ok);
            pause_streaming = false;
            std::thread(&server::rtp::app::start_streaming, this, video_path).detach();
        } 
        else if (method == "GET" && path == "/pause_stream") {
            std::cout << "pause stream" << std::endl;
            send_response(socket , server::http_response::response_type::ok);
            pause_streaming = true;
        }
        else if (method == "GET" && path == "/resume_stream") {
            std::cout << "resume stream" << std::endl;
            send_response(socket , server::http_response::response_type::ok);
            pause_streaming = false;
        }
        else if (method == "GET" && path == "/rewind_stream") {
            std::cout << "rewind stream" << std::endl;
            send_response(socket , server::http_response::response_type::ok);
            rewind_streaming = true;
            pause_streaming = false;
        }
        else if (method == "GET" && path == "/start_detection") {
            std::cout << "detection start" << std::endl;
            send_response(socket , server::http_response::response_type::ok);
            start_detection.store(true);
            
            api::detection::load_model("/Users/gyujinkim/Desktop/Ai/TVM_TUTORIAL/front/yolov5l_m2_raspberry.so");
            std::thread(&api::detection::detect, 
                std::ref(start_detection),
                std::ref(pause_detection),
                std::ref(frame_mutex),
                std::ref(detection_cv),
                std::ref(shared_frame)).detach();
        }
        else if (method == "GET" && path == "/pause_detection") {
            std::cout << "pause detection" << std::endl;
            send_response(socket, server::http_response::response_type::ok);
            pause_detection = true;
        } else if (method == "GET" && path == "/resume_detection") {
            std::cout << "resume detection" << std::endl;
            send_response(socket, server::http_response::response_type::ok);
            pause_detection = false;
            detection_cv.notify_one();
        }
        else {
            send_response(socket , server::http_response::response_type::not_found);
        } 
    } catch (std::exception &e) {
        send_response(socket , server::http_response::response_type::internal_server_error);
    }
}

std::string server::rtp::app::get_gstream_pipeline() const {
    return  "appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
            "rtph264pay config-interval=1 pt=96 ! "
            "udpsink host=192.168.10.121 port=" + std::to_string(port_num) + " auto-multicast=true";           
}

void server::rtp::app::start_streaming(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file" << std::endl;
        return;
    }

    cv::Size frame_size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer;

    if (!writer.open(get_gstream_pipeline(), cv::CAP_GSTREAMER, 0, fps, frame_size)) {
        std::cerr << "Error: Could not open Gstreamer pipeline for writing" << std::endl;
        return;
    }

    cv::Mat frame;
    while (true) {
        // 일시 정지 상태일 경우, 일시 정지 해제될 때까지 대기
        if (pause_streaming) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            if (!frame.empty()) {
                writer.write(frame);  // 현재 프레임을 계속해서 송출하여 정지 상태 유지
            }
            continue;
        }

        if (rewind_streaming) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // 프레임을 처음으로 설정
            rewind_streaming = false;  // 플래그를 초기화
        }

        // 프레임을 읽고 비어있는 경우 영상 끝에 도달했으므로 처음으로 되감기
        cap >> frame;
        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0); 
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            shared_frame = frame.clone();
        }

        writer.write(frame);

        if (start_detection) {
            detection_cv.notify_one();
        }
    }

    cap.release();
    writer.release();
}


void server::rtp::start_server(uint16_t port) {
    boost::asio::io_context io_context;
    boost::asio::ip::tcp::acceptor acceptor(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
    app rtp_app;
    std::cout << "Server started on port " << port << std::endl;
    while (true) {
        boost::asio::ip::tcp::socket socket(io_context);
        acceptor.accept(socket);
        rtp_app.handle_streaming_request(socket);
    }
}