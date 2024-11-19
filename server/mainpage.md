@mainpage Vehicle Monitoring Server API Documentation

# Overview
This documentation provides an overview of the Vehicle Monitoring Server API, detailing its REST endpoints and usage instructions.

---

## 📑 REST API Endpoints

### 🎥 Video Streaming
- **`/start_stream`** : Start RTP video streaming.
- **`/resume_stream`** : Resume RTP video streaming.
- **`/pause_stream`** : Pause RTP video streaming.
- **`/rewind_stream`** : Rewind RTP video streaming.

### 🚗 Vehicle Detection
- **`/start_detection`** : Start vehicle detection.
- **`/pause_detection`** : Pause vehicle detection.
- **`/resume_detection`** : Resume vehicle detection.

---

## 🚀 Getting Started

Follow these steps to install and start the Vehicle Monitoring Server:

1. **Installation**  
   Download and set up the required dependencies using the provided installation script:
   ```bash
   ./install_dependencies.sh
