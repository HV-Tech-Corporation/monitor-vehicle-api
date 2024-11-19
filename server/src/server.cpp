#include "server.hpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>

namespace fs = boost::filesystem;

namespace server {
    namespace rtp {
        std::atomic<bool> pause_streaming(false);  
        std::atomic<bool> rewind_streaming(false); 
        std::atomic<bool> start_detection(false);
        std::atomic<bool> pause_detection(false);

        std::string client_ip;

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

std::string get_content_type(const std::string& file_path) {
    if (file_path.ends_with(".html")) return "text/html";
    if (file_path.ends_with(".css")) return "text/css";
    if (file_path.ends_with(".js")) return "application/javascript";
    if (file_path.ends_with(".png")) return "image/png";
    if (file_path.ends_with(".jpg") || file_path.ends_with(".jpeg")) return "image/jpeg";
    if (file_path.ends_with(".gif")) return "image/gif";
    if (file_path.ends_with(".svg")) return "image/svg+xml";
    if (file_path.ends_with(".woff")) return "font/woff";
    if (file_path.ends_with(".woff2")) return "font/woff2";
    if (file_path.ends_with(".ttf")) return "font/ttf";
    if (file_path.ends_with(".otf")) return "font/otf";
    return "application/octet-stream"; // 기본 값
}

void send_file_doxygen(boost::asio::ip::tcp::socket& socket, const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::string response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        boost::asio::write(socket, boost::asio::buffer(response));
        return;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::string content_type = get_content_type(file_path); // 파일 확장자에 따라 MIME 타입 결정

    std::string response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: " + content_type + "\r\n"; // 동적으로 Content-Type 설정
    response += "Content-Length: " + std::to_string(content.size()) + "\r\n\r\n";
    response += content;

    boost::asio::write(socket, boost::asio::buffer(response));
}


void send_image_list_response(boost::asio::ip::tcp::socket& socket) {
    std::ostringstream response;
    response << "HTTP/1.1 200 OK\r\n";
    response << "Content-Type: text/html\r\n\r\n";

    // HTML 응답 시작 부분
    response << "<html><head><title>Detection Images</title></head><body>";
    response << "<h2>Detection Images</h2>";

    // 이미지 파일들이 저장된 디렉토리
    fs::path directory("/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/build");

    // 디렉토리를 탐색하며 이미지 파일을 HTML에 추가
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (fs::is_regular_file(entry) && 
           (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
            std::string filename = entry.path().filename().string();
            response << "<div><p>" << filename << "</p>";
            response << "<img src='/get_image/" << filename << "' style='max-width:200px;'/></div><br/>";
        }
    }

    response << "</body></html>";

    // HTML 응답 전송
    boost::asio::write(socket, boost::asio::buffer(response.str()));
}

void send_image_response(boost::asio::ip::tcp::socket& socket, const std::string& filename) {
    std::string image_path = "/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/build/" + filename;

    std::ifstream file(image_path, std::ios::binary);
    if (!file) {
        send_response(socket, server::http_response::response_type::not_found);
        return;
    }

    file.seekg(0, std::ios::end);
    std::size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string file_data(file_size, '\0');
    file.read(&file_data[0], file_size);

    std::ostringstream response;
    response << "HTTP/1.1 200 OK\r\n";
    response << "Content-Type: image/jpeg\r\n";
    response << "Content-Length: " << file_size << "\r\n";
    response << "\r\n";

    boost::asio::write(socket, boost::asio::buffer(response.str()));
    boost::asio::write(socket, boost::asio::buffer(file_data));
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
            
            api::detection::load_model("/Users/gyujinkim/Desktop/Ai/TVM_TUTORIAL/front/yolov5n_arm.so");
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
        } 
        else if (method == "GET" && path == "/resume_detection") {
            std::cout << "resume detection" << std::endl;
            send_response(socket, server::http_response::response_type::ok);
            pause_detection = false;
            detection_cv.notify_one();
        } 
        else if (method == "GET" && path == "/get_detection_img") {
            std::cout << "Serving detection images" << std::endl;
            send_image_list_response(socket);
        } 
        else if (method == "GET" && path.find("/show_api_docs") == 0) {
            std::string relative_path = path.substr(strlen("/show_api_docs"));
            if (relative_path.empty() || relative_path == "/") {
                relative_path = "/index.html";
            }

            std::string full_path = "/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/docs/html" + relative_path;
            std::cout << "Serving file: " << full_path << std::endl;
            
            send_file_doxygen(socket, full_path);
            
        }
        else {
            send_response(socket , server::http_response::response_type::not_found);
        } 
    } catch (std::exception &e) {
        send_response(socket , server::http_response::response_type::internal_server_error);
    }
}

std::string server::rtp::app::get_gstream_pipeline() const {
    return  "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
			"jpegenc ! rtpjpegpay ! "
			"udpsink host="+ client_ip + " port=5004 auto-multicast=true";
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

        writer.write(shared_frame);

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
        boost::asio::ip::tcp::endpoint remote_ep = socket.remote_endpoint();
        client_ip = remote_ep.address().to_string();
        rtp_app.handle_streaming_request(socket);
    }
}