#include "server.hpp"

int main() {
    server::rtp::start_server(8080);
    return 0;
}
