#include "response.hpp"

namespace server {
    namespace http_response {
        namespace status_strings {
            const std::string ok = "HTTP/1.1 200 OK\r\n";
            const std::string bad_request = "HTTP/1.0 400 Bad Request\r\n";
            const std::string unauthorized = "HTTP/1.0 401 Unauthorized\r\n";
            const std::string forbidden = "HTTP/1.0 403 Forbidden\r\n";
            const std::string not_found = "HTTP/1.0 404 Not Found\r\n";
            const std::string internal_server_error = "HTTP/1.0 500 Internal Server Error\r\n";

            const std::string& to_string(response_type status) {
                switch (status) {
                    case response_type::ok: return ok;
                    case response_type::bad_request: return bad_request;
                    case response_type::unauthorized: return unauthorized;
                    case response_type::forbidden: return forbidden;
                    case response_type::not_found: return not_found;
                    case response_type::internal_server_error: return internal_server_error;
                    default: return internal_server_error;
                }
            }
        } //namespace status_string

        namespace misc_strings {
            const std::string name_value_separator = ": ";
            const std::string crlf = "\r\n";
        } //namespace misc_strings

    
        // 상태 코드 설정 함수
        void response::set_status(response_type status_code) {
            status = status_code;
        }

        // 헤더 추가 함수
        void response::add_header(const std::string& name, const std::string& value) {
            headers.emplace_back(name, value);
        }

        // 응답을 버퍼로 변환하는 함수
        std::vector<boost::asio::const_buffer> response::to_buffers() const {
            std::vector<boost::asio::const_buffer> buffers;

            // 상태 문자열 추가
            buffers.push_back(boost::asio::buffer(status_strings::to_string(status)));

            // 헤더 추가
            for (const auto& header : headers) {
                buffers.push_back(boost::asio::buffer(header.first));
                buffers.push_back(boost::asio::buffer(misc_strings::name_value_separator));
                buffers.push_back(boost::asio::buffer(header.second));
                buffers.push_back(boost::asio::buffer(misc_strings::crlf));
            }

            // 헤더와 본문 사이 줄바꿈 추가
            buffers.push_back(boost::asio::buffer(misc_strings::crlf));

            // 본문 추가
            buffers.push_back(boost::asio::buffer(content));

            return buffers;
        }

        namespace response_body {
            const char ok[] = "";
            const char bad_requeset[] = 
                "<html>"
                "<head><title>Bad Request</title></head>"
                "<body><h1>400 Bad Request</h1></body>"
                "</html>";
            const char unauthorized[] = 
                "<html>"
                "<head><title>Unauthorized</title></head>"
                "<body><h1>401 Unauthorized</h1></body>"
                "</html>";
            const char forbidden[] = 
                "<html>"
                "<head><title>Forbidden</title></head>"
                "<body><h1>403 Forbidden</h1></body>"
                "</html>";
            const char not_found[] = 
                "<html>"
                "<head><title>Not Found</title></head>"
                "<body><h1>404 Not Found</h1></body>"
                "</html>";
            const char internal_server_error[] = 
                "<html>"
                "<head><title>InternalServerError</title></head>"
                "<body><h1>500 Internal Server Error</h1></body>"
                "</html>";
        }

        std::string response::response_body_to_string(response_type status) {
            switch (status)
            {
                case response_type::ok: 
                    return response_body::ok;
                case response_type::bad_request:  
                    return response_body::bad_requeset;
                case response_type::unauthorized:
                    return response_body::unauthorized;
                case response_type::forbidden:
                    return response_body::forbidden;
                case response_type::not_found:
                    return response_body::not_found;
                case response_type::internal_server_error:
                    return response_body::internal_server_error;
            
            default:
                break;
            }
        }

    } // namespace http_response
} // namespace server
