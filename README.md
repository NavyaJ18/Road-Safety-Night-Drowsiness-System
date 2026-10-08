# 🚗 Driver Drowsiness Detection and Road Safety Monitoring System

A real-time computer vision-based system designed to monitor driver alertness and identify potential signs of drowsiness.

The system analyzes facial features through a live camera feed and calculates multiple behavioural parameters associated with driver fatigue. A Streamlit-based dashboard is used for visualization, monitoring, and data logging.

---

## 📌 Project Overview

Driver drowsiness is an important road-safety concern, particularly during long-distance and night-time driving. Reduced alertness can affect attention and driving performance.

This project focuses on detecting behavioural indicators of drowsiness using facial landmark analysis and continuously monitoring relevant parameters.

The current implementation combines computer vision, data logging, dashboard visualization, and a prototype UDP-based sensor communication layer.

---

## 🎯 Project Objectives

- Detect potential signs of driver drowsiness using computer vision.
- Monitor eye and mouth-related facial parameters.
- Analyze prolonged eye closure using PERCLOS.
- Monitor blink behaviour.
- Detect yawning behaviour.
- Record detected parameters with timestamps.
- Visualize recorded parameters through an interactive dashboard.
- Provide a foundation for integration with additional sensors.
- Develop a prototype for network-based sensor data acquisition.
- Monitor communication parameters such as latency and packet loss.

---

## 🔍 Parameters Used

The current system monitors the following parameters:

- 👁️ Eye Aspect Ratio (EAR)
- 😮 Mouth Aspect Ratio (MAR)
- ⏱️ PERCLOS
- 👀 Eye-state classification
- 👁️ Blink rate
- 🥱 Yawning status
- 🧠 Overall drowsiness status
- 🕒 Timestamp information

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Computer Vision

- OpenCV
- MediaPipe Face Landmarker

### Data Processing

- Pandas
- Tab-separated data logging

### Dashboard

- Streamlit

### Communication

- UDP sockets
- JSON-based sensor packets

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

## 📊 System Overview

The system consists of two main monitoring components.

### 🧠 Drowsiness Detection

The computer-vision pipeline processes a live camera feed and extracts facial landmarks to calculate:

```text
Webcam
   ↓
Face Landmark Detection
   ↓
Eye and Mouth Landmarks
   ↓
EAR + MAR
   ↓
Eye State + Yawning
   ↓
Blink Rate + PERCLOS
   ↓
Drowsiness Status
   ↓
Data Logging
   ↓
Dashboard

### 📡 Sensor Network

The system also includes a prototype distributed sensor network for acquiring additional physiological, vehicle, motion, GPS, and environmental parameters.

The sensor network is designed around multiple ESP32-S3 nodes connected to a central Raspberry Pi 5.

```text
Physiological Sensors
        ↓
   ESP32-S3 #1
        │
        │
Vehicle / Motion Sensors
        ↓
   ESP32-S3 #2
        │
        │
GPS / Environmental Sensors
        ↓
   ESP32-S3 #3
        │
        │
        └──────────────┐
                       ↓
                 Wi-Fi / UDP
                       ↓
                Raspberry Pi 5
                       ↓
              Data Aggregation
                       ↓
          Validation & Synchronization
                       ↓
                Sensor Fusion
                       ↓
              Drowsiness Analysis
                       ↓
                  Dashboard
