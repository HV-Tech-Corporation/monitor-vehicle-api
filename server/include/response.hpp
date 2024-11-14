#ifndef RESPONSE_HPP
#define RESPONSE_HPP

#include <string>
#include <vector>
#include <boost/asio.hpp>

namespace server {
    namespace http_response {

        // 응답 상태 코드
        enum class response_type {
            ok = 200,
            bad_request = 400,
            unauthorized = 401,
            forbidden = 403,
            not_found = 404,
            internal_server_error = 500,
        };

        // 응답 구조체 정의
        struct response {
            response_type status;
            std::vector<std::pair<std::string, std::string>> headers;
            std::string content;

            // 상태 코드 설정
            void set_status(response_type status_code);

            // 헤더 추가
            void add_header(const std::string& name, const std::string& value);

            // 버퍼로 변환하여 네트워크 전송 준비
            std::vector<boost::asio::const_buffer> to_buffers() const;
            std::string response_body_to_string(response_type status);
        };

    } // namespace http_response
} // namespace server

#endif
