#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <iostream>
#include <string>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>


void handle_client(int client_socket) {
    #include <iostream>
#include <boost/asio.hpp>
#include "response.hpp"

using boost::asio::ip::tcp;
namespace http = server::http_response;

void handle_client(tcp::socket& socket) {
    try {
        // 클라이언트로부터 요청 수신 (단순화된 HTTP 요청 수신)
        char data[1024];
        boost::system::error_code error;
        size_t length = socket.read_some(boost::asio::buffer(data), error);

        if (error == boost::asio::error::eof) {
            // 연결 종료
            return;
        } else if (error) {
            throw boost::system::system_error(error);
        }

        std::string request(data, length);
        std::cout << "Received request: " << request << std::endl;

        // 응답 객체 생성 및 설정
        http::response res;
        
        // 요청 경로에 따라 응답 설정
        if (request.find("GET / ") != std::string::npos) {
            res.set_status(http::response_type::ok);
            res.add_header("Content-Type", "text/plain");
            res.content = "Hello, World!";
        } else {
            res.set_status(http::response_type::not_found);
            res.add_header("Content-Type", "text/plain");
            res.content = "404 Not Found";
        }

        // 응답 버퍼 전송
        auto buffers = res.to_buffers();
        boost::asio::write(socket, buffers);
    }
    catch (std::exception& e) {
        std::cerr << "Exception in client handling: " << e.what() << std::endl;
    }
}

void start_server(unsigned short port) {
    boost::asio::io_context io_context;
    tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));

    std::cout << "Server started on port " << port << std::endl;

    while (true) {
        tcp::socket socket(io_context);
        acceptor.accept(socket);

        // 클라이언트 요청 처리
        handle_client(socket);
    }
}

int main() {
    try {
        start_server(8080);  // 8080 포트에서 서버 시작
    }
    catch (std::exception& e) {
        std::cerr << "Server exception: " << e.what() << std::endl;
    }
    return 0;
}

}

int main() {
    // 비디오 파일 열기
    cv::VideoCapture cap("HDCCTVTraffic.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    // 비디오 크기 정보 가져오기
    cv::Size frame_size = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // 비디오 FPS 가져오기
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "FPS: " << fps << std::endl;

    // GStreamer 파이프라인 설정
    std::string pipeline = "appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
                           "rtph264pay config-interval=1 pt=96 ! "
                           "udpsink host=127.0.0.1 port=5004";

    // GStreamer 파이프라인을 통해 VideoWriter 열기
    cv::VideoWriter writer;
    if (!writer.open(pipeline, cv::CAP_GSTREAMER, 0, fps, frame_size, true)) {
        std::cerr << "Error: Could not open GStreamer pipeline for writing." << std::endl;
        return -1;
    }

    cv::Mat frame;
    double start_time = cv::getTickCount(); // 시작 시간

    while (true) {
        // 비디오 파일에서 프레임 읽기
        cap >> frame;
        if (frame.empty()) break;  // 파일 끝에 도달하면 종료

        // 현재 타임스탬프 계산 (초 단위)
        double time_in_seconds = (cv::getTickCount() - start_time) / cv::getTickFrequency();
        
        // 타임스탬프를 비디오 프레임에 추가
        std::stringstream timestamp_stream;
        timestamp_stream << "Time: " << time_in_seconds << "s";
        std::string timestamp = timestamp_stream.str();

        // 텍스트로 타임스탬프 표시
        cv::putText(frame, timestamp, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

        // 프레임을 GStreamer를 통해 전송
        writer.write(frame);

        // 프레임 간 시간 지연 (FPS에 맞게 지연)
        double elapsed_time = (cv::getTickCount() - start_time) / cv::getTickFrequency();
        double delay_time = (1.0 / fps) - elapsed_time;
        if (delay_time > 0) {
            cv::waitKey(delay_time * 1000);  // 밀리초로 변환
        }
    }

    // 리소스 해제
    cap.release();
    writer.release();
    return 0;
}
