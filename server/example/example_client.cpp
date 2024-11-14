#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

void receive_rtp_stream() {
    // GStreamer RTP 스트림을 수신하는 파이프라인
    std::string pipeline = "udpsrc port=5004 ! application/x-rtp, payload=96 ! "
                           "rtph264depay ! avdec_h264 ! videoconvert ! appsink";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open RTP stream for reading" << std::endl;
        return;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame received from RTP stream" << std::endl;
            break;
        }

        // 프레임을 화면에 출력
        cv::imshow("RTP Stream", frame);
        if (cv::waitKey(30) >= 0) break;  // 아무 키나 누르면 종료
    }

    cap.release();
    cv::destroyAllWindows();
}

void send_request(const std::string& server_ip, uint16_t server_port) {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        std::cerr << "Error: Could not create socket" << std::endl;
        return;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);

    // 서버 IP 주소 설정
    if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Error: Invalid address" << std::endl;
        close(client_socket);
        return;
    }

    // 서버에 연결
    if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error: Connection failed" << std::endl;
        close(client_socket);
        return;
    }

    // GET 요청 생성
    std::string request = std::string("GET ") + "/start_stream" + " HTTP/1.1\r\nHost: " + server_ip + "\r\n\r\n";

    // 요청 전송
    send(client_socket, request.c_str(), request.size(), 0);

    // 응답 수신
    char buffer[1024] = {0};
    int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received > 0) {
        buffer[bytes_received] = '\0';
        std::cout << "Response from server:\n" << buffer << std::endl;

        // 응답 수신 후 RTP 스트림을 수신하는 함수 호출
        receive_rtp_stream();
    } else {
        std::cerr << "Error: No response from server" << std::endl;
    }

    // 소켓 닫기
    close(client_socket);
}

int main() {
    std::string server_ip = "127.0.0.1";  // 서버 IP
    uint16_t server_port = 8080;          // 서버 포트
    // std::string request_path = "/start_stream";  // 요청 경로

    send_request(server_ip, server_port);

    return 0;
}
