#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>

// 서버에 GET 요청 보내기
bool send_start_stream_request(const std::string& server_ip, uint16_t server_port) {
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        std::cerr << "Error: Could not create socket" << std::endl;
        return false;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);

    // 서버 IP 주소 설정
    if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) <= 0) {
        std::cerr << "Error: Invalid address" << std::endl;
        close(client_socket);
        return false;
    }

    // 서버에 연결
    if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error: Connection failed" << std::endl;
        close(client_socket);
        return false;
    }

    // GET 요청 생성
    std::string request = "GET /start_stream HTTP/1.1\r\nHost: " + server_ip + "\r\n\r\n";

    // 요청 전송
    send(client_socket, request.c_str(), request.size(), 0);

    // 응답 수신
    char buffer[1024] = {0};
    int bytes_received = recv(client_socket, buffer, sizeof(buffer) - 1, 0);
    close(client_socket);

    if (bytes_received > 0) {
        buffer[bytes_received] = '\0';
        std::cout << "Response from server:\n" << buffer << std::endl;
        return true;
    } else {
        std::cerr << "Error: No response from server" << std::endl;
        return false;
    }
}

// RTP 스트림 수신
void receive_rtp_stream() {
    std::string pipeline = "udpsrc port=5004 caps=\"application/x-rtp, payload=96\" ! "
                           "rtpjpegdepay ! queue ! jpegdec ! queue ! "
                           "videoconvert ! appsink";

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

        cv::imshow("RTP Stream", frame);
        if (cv::waitKey(30) >= 0) break;  // 아무 키나 누르면 종료
    }

    cap.release();
    cv::destroyAllWindows();
}

int main() {
    std::string server_ip = "127.0.0.1";
    uint16_t server_port = 8080;

    // 서버에 /start_stream 요청
    if (!send_start_stream_request(server_ip, server_port)) {
        std::cerr << "Failed to start stream on server." << std::endl;
        return -1;
    }

    // RTP 스트림 수신 시작
    receive_rtp_stream();

    return 0;
}
