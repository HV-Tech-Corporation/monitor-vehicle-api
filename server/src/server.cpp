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
        std::atomic<bool> shared_frame_updated(false);
        std::atomic<bool> is_streaming(false);

        std::atomic<bool> preload_complete(false);
        
        std::unordered_map<int, cv::Mat> detected_frames;
        std::mutex detected_frames_mutex;
        std::mutex socket_mutex;

        std::mutex bestShot_mutex;
        cv::Mat bestShot_frame;

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
    res_inst.add_header("Connection", "keep-alive");

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

void send_multipart_image_response(boost::asio::ip::tcp::socket& socket, const std::vector<cv::Mat>& images) {
    std::cout << "multipart process executed..." << std::endl;
    std::string boundary = "--frame";

    // HTTP 응답 헤더 작성
    std::ostringstream header;
    header << "HTTP/1.1 400 Bad Request\r\n"
           << "Content-Type: multipart/mixed; boundary=" << boundary << "\r\n"
           << "Connection: keep-alive\r\n"
           << "\r\n";

    try {
        // 헤더 전송
        boost::asio::write(socket, boost::asio::buffer(header.str()));

        for (const auto& image : images) {
            std::vector<uchar> buf;
            cv::imencode(".jpg", image, buf);  // 이미지를 JPEG로 인코딩

            // 각 파트의 시작
            std::ostringstream part;
            part << "--" << boundary << "\r\n"
                 << "Content-Type: image/jpeg\r\n"
                 << "Content-Length: " << buf.size() << "\r\n"
                 << "\r\n";
            boost::asio::write(socket, boost::asio::buffer(part.str()));
            boost::asio::write(socket, boost::asio::buffer(buf));
            boost::asio::write(socket, boost::asio::buffer("\r\n"));
        }

        // 멀티파트 종료
        std::ostringstream end_boundary;
        end_boundary << "--" << boundary << "--\r\n";
        boost::asio::write(socket, boost::asio::buffer(end_boundary.str()));

    } catch (const boost::system::system_error& e) {
        std::cerr << "Error sending multipart image: " << e.what() << std::endl;
    }
}


void server::rtp::app::handle_streaming_request(boost::asio::ip::tcp::socket& socket) {
    
    try {
        boost::asio::streambuf request;
        boost::asio::read_until(socket, request, "\r\n");

        std::istream request_stream(&request);
        std::string method, path;
        request_stream >> method >> path;

        if (method == "GET" && path == "/start_stream") {
            if (is_streaming.load()) {
                send_response(socket, server::http_response::response_type::bad_request);
                return;
            }

            send_response(socket, server::http_response::response_type::ok); // 먼저 응답을 보냄

            is_streaming.store(true);
            pause_streaming = false;

            std::string video_path = "/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/traffic_jam2.mp4";

            // 새로운 스트리밍 쓰레드 생성
            std::thread([this, socket = std::move(socket), video_path] () mutable {
                try {
                    start_streaming(video_path, socket);
                } catch (const std::exception& e) {
                    std::cerr << "Error in streaming thread: " << e.what() << std::endl;
                }
                is_streaming.store(false); // 스트리밍 종료 시 상태 업데이트
            }).detach();

            std::cout << "start stream" << std::endl;
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
            send_response(socket, server::http_response::response_type::ok);
            start_detection.store(true);
            detection_cv.notify_one();
            
            api::detection::load_model("/Users/gyujinkim/Desktop/Ai/TVM_TUTORIAL/front/yolov5n_m2_raspberry.so");
            std::thread(&api::detection::detect, 
                        std::ref(start_detection),
                        std::ref(pause_detection),
                        std::ref(shared_frame_updated),
                        std::ref(frame_mutex),
                        std::ref(detection_cv),
                        std::ref(shared_frame)).detach();
        }
        else if (method == "GET" && path == "/preprocess_detection") {
            send_response(socket, server::http_response::response_type::ok);
            api::detection::load_model("/Users/gyujinkim/Desktop/Ai/TVM_TUTORIAL/front/yolov5n_m2_raspberry.so");
            std::thread detection_thread([&]() {
                api::detection::preprocess_detect(
                    "/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/traffic_jam2.mp4",
                    std::ref(detected_frames),
                    std::ref(detected_frames_mutex),
                    std::ref(preload_complete),
                    std::ref(bestShot_mutex),
                    std::ref(bestShot_frame)
                );
            });
            detection_thread.detach(); // 스레드를 분리하여 서버를 차단하지 않음
        }
        else if (method == "GET" && path == "/pause_detection") {
            std::cout << "pause detection" << std::endl;
            start_detection.store(false);
            send_response(socket, server::http_response::response_type::ok);
            // pause_detection = true;
        } 
        else if (method == "GET" && path == "/resume_detection") {
            std::cout << "resume detection" << std::endl;
            send_response(socket, server::http_response::response_type::ok);
            start_detection.store(true);
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

void server::rtp::app::start_streaming(const std::string& video_path, boost::asio::ip::tcp::socket& socket) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file" << std::endl;
        return;
    }

    cv::Size frame_size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "fps is : " << fps << std::endl;

    int frame_delay_ms = static_cast<int>(1000.0 / fps); // 프레임 간 대기 시간

    cv::VideoWriter writer;
    if (!writer.open(get_gstream_pipeline(), cv::CAP_GSTREAMER, 0, fps, frame_size)) {
        std::cerr << "Error: Could not open Gstreamer pipeline for writing" << std::endl;
        return;
    }

    int frame_counter = 0; // 프레임 번호
    cv::Mat frame;

    while (true) {
        auto start_time = std::chrono::steady_clock::now(); // 루프 시작 시간

        if (pause_streaming) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        if (rewind_streaming) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            frame_counter = 0;
            rewind_streaming = false;
        }

        cap >> frame; // 원본 프레임 읽기
        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            frame_counter = 0;
            continue;
        }

        cv::Mat output_frame;
        if (preload_complete.load()) {
            // 탐지된 프레임 사용
            std::lock_guard<std::mutex> lock(detected_frames_mutex);
            if (detected_frames.find(frame_counter) != detected_frames.end()) {
                output_frame = detected_frames[frame_counter]; // 탐지된 프레임 사용
                for (const auto& entry : std::filesystem::directory_iterator(".")) {
                    std::string file_name = entry.path().filename().string();
                    // 파일 이름이 "bestShot_<frame_index>_<object_id>.jpg" 형식인지 확인
                    std::string prefix = "bestShot_" + std::to_string(frame_counter) + "_";
                    if (file_name.find(prefix) == 0 && file_name.substr(file_name.size() - 4) == ".jpg") {
                        cv::Mat bestShot_frame2 = cv::imread("/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/bestShot_2_3.jpg");
                        if (!bestShot_frame2.empty()) {
                            // 각 이미지를 클라이언트에 전송
                            send_multipart_image_response(socket, bestShot_frame2);
                        }
                    }
                }
            } else {
                output_frame = frame.clone(); // 탐지 결과가 없을 경우 원본 사용
            }
        } else {
            output_frame = frame.clone(); // 탐지 결과가 준비되지 않으면 원본 사용
        }

        writer.write(output_frame); // 프레임 송출
        frame_counter++;

        // 전송 속도 제어
        auto end_time = std::chrono::steady_clock::now();
        int elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        int sleep_time = frame_delay_ms - elapsed_ms;
        if (sleep_time > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
        }
    }

    cap.release();
    writer.release();
}

void enable_tcp_keep_alive(boost::asio::ip::tcp::socket& socket) {
    boost::asio::socket_base::keep_alive option(true);
    socket.set_option(option);  // TCP Keep-Alive 활성화
}

void server::rtp::start_server(uint16_t port) {
    boost::asio::io_context io_context;
    boost::asio::ip::tcp::acceptor acceptor(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
    app rtp_app;
    std::vector<std::thread> thread_pool;

    std::cout << "Server started on port " << port << std::endl;

    while (true) {
        boost::asio::ip::tcp::socket socket(io_context);
        acceptor.accept(socket);

        try {
            if (socket.is_open()) {
                boost::asio::socket_base::keep_alive option(true);
                socket.set_option(option);

                boost::asio::ip::tcp::endpoint remote_ep = socket.remote_endpoint();
                client_ip = remote_ep.address().to_string();

                // 새로운 스레드 생성
                thread_pool.emplace_back([&rtp_app, socket = std::move(socket)]() mutable {
                    try {
                        rtp_app.handle_streaming_request(socket);
                    } catch (const std::exception& e) {
                        std::cerr << "Error in client thread: " << e.what() << std::endl;
                    }
                });

                // 스레드 관리 (주기적으로 정리)
                for (auto it = thread_pool.begin(); it != thread_pool.end();) {
                    if (it->joinable()) {
                        it->join();
                        it = thread_pool.erase(it);
                    } else {
                        ++it;
                    }
                }
            } else {
                std::cerr << "Error: Socket is not open after accept" << std::endl;
            }
        } catch (const boost::system::system_error& e) {
            std::cerr << "Error handling client: " << e.what() << std::endl;
        }
    }
}
