#include "server.hpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <sstream>

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
        std::unordered_map<int, int> kruskal_results_per_frames;

        std::mutex detected_frames_mutex;
        std::mutex kruskal_results_per_frames_mutex;
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

void send_response(std::shared_ptr<boost::asio::ip::tcp::socket> shared_socket, server::http_response::response_type status) {
    server::http_response::response res_inst;
    server::http_response::response _http_response;
    res_inst.set_status(status);
    res_inst.add_header("Content-Type", "text/html");
    res_inst.add_header("Connection", "keep-alive");
    // 응답 본문을 설정 (response_type에 따라 본문 선택)
    res_inst.content = _http_response.response_body_to_string(status);
    auto buffers = res_inst.to_buffers();
    boost::asio::write(*shared_socket, buffers);
}

void send_single_image_response(std::shared_ptr<boost::asio::ip::tcp::socket> shared_socket, const cv::Mat& image, int class_id, int kruskal_results_per_frames) {
    // std::cout << "Sending single image with XML description..." << std::endl;

    if (image.empty()) {
        std::cerr << "Error: Input image is empty." << std::endl;
        return;
    }

    // 이미지 인코딩 (JPEG 형식)
    std::vector<uchar> buf;
    if (!cv::imencode(".jpg", image, buf)) {
        std::cerr << "Error: Failed to encode image." << std::endl;
        return;
    }

    if (buf.empty()) {
        std::cerr << "Error: Encoded buffer is empty." << std::endl;
        return;
    }

    if (!shared_socket || !shared_socket->is_open()) {
        std::cerr << "Error: Socket is not open or is null." << std::endl;
        return;
    }

    // 현재 시간을 가져오기
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream time_stream;
    time_stream << std::put_time(std::localtime(&now_time_t), "%Y-%m-%dT%H:%M:%S");
    
    std::string class_name;
    if (class_id == 1) class_name = "bicycle";
    else if (class_id == 2) class_name = "car";
    else if (class_id == 3) class_name = "motorbike";
    else if (class_id == 5) class_name = "bus";
    else {
        std::cerr << "Error: Invalid class_id: " << class_id << std::endl;
        class_name = "unknown";
    }

    // XML 데이터 생성
    std::ostringstream xml_description;
    xml_description << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                    << "<response>\n"
                    << "  <description>Image with timestamp</description>\n"
                    << "  <costs>"<< std::to_string(kruskal_results_per_frames) << "</costs>\n"
                    << "  <class>"<< class_name <<"</class>\n"
                    << "  <timestamp>" << time_stream.str() << "</timestamp>\n"
                    << "</response>\n";

    std::string xml_data = xml_description.str();
    size_t xml_size = xml_data.size();

    // HTTP 응답 헤더 작성
    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: multipart/related; boundary=--boundary\r\n"
           << "Connection: close\r\n"
           << "\r\n";

    // 멀티파트 데이터 작성
    std::ostringstream multipart;
    multipart << "--boundary\r\n"
              << "Content-Type: application/xml\r\n"
              << "Content-Length: " << xml_size << "\r\n\r\n"
              << xml_data << "\r\n"
              << "--boundary\r\n"
              << "Content-Type: image/jpeg\r\n"
              << "Content-Length: " << buf.size() << "\r\n\r\n";

    std::string header_str = header.str();
    std::string multipart_str = multipart.str();

    try {
        // 헤더 전송
        boost::asio::write(*shared_socket, boost::asio::buffer(header_str));

        // 멀티파트 헤더 및 XML 데이터 전송
        boost::asio::write(*shared_socket, boost::asio::buffer(multipart_str));
        boost::asio::write(*shared_socket, boost::asio::buffer(xml_data));

        // 이미지 데이터 전송
        size_t total_bytes_sent = 0;
        while (total_bytes_sent < buf.size()) {
            size_t bytes_sent = boost::asio::write(
                *shared_socket, boost::asio::buffer(buf.data() + total_bytes_sent, buf.size() - total_bytes_sent));
            total_bytes_sent += bytes_sent;
        }

        // 멀티파트 종료
        std::string boundary_end = "\r\n--boundary--\r\n";
        boost::asio::write(*shared_socket, boost::asio::buffer(boundary_end));

        std::cout << "Successfully sent XML and image data." << std::endl;
    } catch (const boost::system::system_error& e) {
        if (e.code() == boost::asio::error::broken_pipe) {
            std::cerr << "Broken pipe: Client disconnected prematurely." << std::endl;
        } else {
            std::cerr << "Error sending response: " << e.what() << std::endl;
        }
    }
}

void server::rtp::app::handle_streaming_request(std::shared_ptr<boost::asio::ip::tcp::socket> shared_socket) {
    
    try {
        boost::asio::streambuf request;
        boost::asio::read_until(*shared_socket, request, "\r\n");

        std::istream request_stream(&request);
        std::string method, path;
        request_stream >> method >> path;
        if (method == "GET" && path == "/start_stream") {
            if (is_streaming.load()) {
                send_response(shared_socket, server::http_response::response_type::bad_request);
                return;
            }
   
            send_response(shared_socket, server::http_response::response_type::ok); // 먼저 응답을 보냄
            auto streaming_socket = std::make_shared<boost::asio::ip::tcp::socket>(std::move(*shared_socket));

            is_streaming.store(true);
            pause_streaming = false;

            std::string video_path = "/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/traffic_jam2.mp4";
            
            // 새로운 스트리밍 쓰레드 생성
            std::thread([this, streaming_socket, video_path]() mutable {
                try {
                    start_streaming(video_path, streaming_socket);
                } catch (const std::exception& e) {
                    std::cerr << "Error in streaming thread: " << e.what() << std::endl;
                }
                is_streaming.store(false); // 스트리밍 종료 시 상태 업데이트
            }).detach();
            std::cout << "start stream" << std::endl;
        }

        else if (method == "GET" && path == "/pause_stream") {
            std::cout << "pause stream" << std::endl;
            send_response(shared_socket , server::http_response::response_type::ok);
            pause_streaming = true;
        }
        else if (method == "GET" && path == "/resume_stream") {
            std::cout << "resume stream" << std::endl;
            send_response(shared_socket , server::http_response::response_type::ok);
            pause_streaming = false;
        }
        else if (method == "GET" && path == "/rewind_stream") {
            std::cout << "rewind stream" << std::endl;
            send_response(shared_socket , server::http_response::response_type::ok);
            rewind_streaming = true;
            pause_streaming = false;
        }
        else if (method == "GET" && path == "/start_detection") {
            send_response(shared_socket, server::http_response::response_type::ok);
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
            send_response(shared_socket, server::http_response::response_type::ok);
            api::detection::load_model("/Users/gyujinkim/Desktop/Ai/TVM_TUTORIAL/front/yolov5n_m2_raspberry.so");
            std::thread detection_thread([&]() {
                api::detection::preprocess_detect(
                    "/Users/gyujinkim/Desktop/Github/monitor-vehicle-api/server/traffic_jam2.mp4",
                    std::ref(detected_frames),
                    std::ref(detected_frames_mutex),
                    std::ref(kruskal_results_per_frames),
                    std::ref(kruskal_results_per_frames_mutex),
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
            send_response(shared_socket, server::http_response::response_type::ok);
        } 
        else if (method == "GET" && path == "/resume_detection") {
            std::cout << "resume detection" << std::endl;
            send_response(shared_socket, server::http_response::response_type::ok);
            start_detection.store(true);
        } 
        else {
            send_response(shared_socket , server::http_response::response_type::not_found);
        } 
    } catch (std::exception &e) {
        send_response(shared_socket , server::http_response::response_type::internal_server_error);
    }
}

std::string server::rtp::app::get_gstream_pipeline() const {
    return  "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
			"jpegenc ! rtpjpegpay ! "
			"udpsink host="+ client_ip + " port=5004 auto-multicast=true";
}

void server::rtp::app::start_streaming(const std::string& video_path, std::shared_ptr<boost::asio::ip::tcp::socket> shared_socket) {
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
            std::lock_guard<std::mutex> lock(detected_frames_mutex);
            if (detected_frames.find(frame_counter) != detected_frames.end()) {
                output_frame = detected_frames[frame_counter]; // 탐지된 프레임 사용
                for (const auto& entry : std::filesystem::directory_iterator(".")) {
                    std::string file_name = entry.path().filename().string();
                    // 파일 이름이 "bestShot_<frame_index>_<object_id>.jpg" 형식인지 확인
                    std::string prefix = "bestShot_" + std::to_string(frame_counter) + "_";
                    if (file_name.find(prefix) == 0 && file_name.substr(file_name.size() - 4) == ".jpg") {
                        // "class_ids_" 뒤의 값을 찾기 위해 정규 표현식 또는 단순 검색 사용
                        std::string class_id_prefix = "class_ids_";
                        size_t class_id_pos = file_name.find(class_id_prefix);
                        if (class_id_pos != std::string::npos) {
                            size_t class_id_start = class_id_pos + class_id_prefix.size();
                            size_t class_id_end = file_name.find('_', class_id_start); // 다른 구분자가 있을 경우 처리
                            std::string class_id_str = file_name.substr(class_id_start, class_id_end - class_id_start);
                            
                            // class_ids 인덱스를 정수로 변환
                            int class_id = std::stoi(class_id_str);

                            cv::Mat bestShot_frame2 = cv::imread(entry.path().string());
                            if (!bestShot_frame2.empty()) {
                                // 클래스 ID와 이미지를 함께 클라이언트로 전송
                                send_single_image_response(shared_socket, bestShot_frame2, class_id, kruskal_results_per_frames[frame_counter]);
                            }
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


// Initialize the io_context object for asynchronous I/O operations.
// Infinite loop to continuously accept incoming client connections.
void server::rtp::start_server(uint16_t port) {
    
    boost::asio::io_context io_context;
    boost::asio::ip::tcp::acceptor acceptor(io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port));
    app rtp_app;

    std::cout << "Server started on port " << port << std::endl;

    while (true) {
        auto shared_socket = std::make_shared<boost::asio::ip::tcp::socket>(io_context);
        acceptor.accept(*shared_socket);

        try {
            if (shared_socket->is_open()) {
                boost::asio::socket_base::keep_alive option(true);
                shared_socket->set_option(option);

                boost::asio::ip::tcp::endpoint remote_ep = shared_socket->remote_endpoint();
                client_ip = remote_ep.address().to_string();

                std::thread([&rtp_app, shared_socket]() {
                    try {
                        rtp_app.handle_streaming_request(shared_socket);
                    } catch (const std::exception& e) {
                        std::cerr << "Error in client thread: " << e.what() << std::endl;
                    }
                }).detach();
            } else {
                std::cerr << "Error: Socket is not open after accept" << std::endl;
            }
        } catch (const boost::system::system_error& e) {
            std::cerr << "Error handling client: " << e.what() << std::endl;
        }
    }
}

