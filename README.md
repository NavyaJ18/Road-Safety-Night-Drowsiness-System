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



## 📡 Sensor Network Architecture

The sensor network is organized into three ESP32-S3 nodes based on the type and physical location of the sensors.

This distributed arrangement reduces unnecessary wiring and allows different groups of sensors to be acquired locally before transmitting their measurements to the central processing unit.

### 🧠 ESP32-S3 Node 1 — Physiological Monitoring

The first ESP32-S3 node is responsible for acquiring physiological parameters related to driver state.

| Sensor | Parameter | Interface |
|---|---|---|
| MAX30102 | Heart rate and SpO₂ | I²C |
| GSR / EDA | Skin conductance | Analog |
| AD8232 | ECG signal | Analog |

The node performs local sensor acquisition and transmits the collected measurements to the Raspberry Pi through the wireless network.

```text
MAX30102 ──┐
           │
GSR / EDA ─┼──→ ESP32-S3 #1 ──→ Wi-Fi / UDP
           │
AD8232 ────┘
---

## 🔧 Hardware and Sensor Specifications

The prototype hardware is organized into three ESP32-S3 sensor nodes and one Raspberry Pi 5 central processing unit. Each node is assigned a specific group of sensors based on their function and physical placement.

### 🧩 Core Processing Hardware

| Hardware | Quantity | Purpose |
|---|---:|---|
| ESP32-S3-DevKitC-1 N8R8 | 3 | Distributed sensor acquisition and wireless communication |
| Raspberry Pi 5 8GB | 1 | Central data aggregation, processing, synchronization and sensor fusion |
| IR / Night-Vision USB Camera | 1 | Driver facial monitoring under low-light conditions |
| IR Illuminator | 1 | Additional illumination for night-time facial monitoring |

### 🧠 Physiological Sensors

These sensors are connected to **ESP32-S3 Node 1**.

| Sensor | Parameter | Interface | Purpose |
|---|---|---|---|
| MAX30102 | Heart rate and SpO₂ | I²C | Monitor cardiovascular and oxygen-related parameters |
| GSR / EDA Sensor | Skin conductance | Analog | Monitor changes in electrodermal activity |
| AD8232 | ECG signal | Analog | Acquire ECG-related electrical activity |

These measurements provide additional physiological information that can complement the computer-vision-based drowsiness indicators.

### 🚗 Vehicle and Motion Sensors

These components are connected to **ESP32-S3 Node 2**.

| Sensor / Interface | Parameter | Interface | Purpose |
|---|---|---|---|
| MPU6050 | Acceleration and angular velocity | I²C | Monitor vehicle motion, vibration and movement |
| OBD-II / CAN Interface | Vehicle parameters | CAN / serial interface | Acquire supported vehicle information from the vehicle network |

The availability of OBD-II parameters depends on the specific vehicle and its supported diagnostic data.

### 📍 GPS and Environmental Sensors

These sensors are connected to **ESP32-S3 Node 3**.

| Sensor | Parameter | Interface | Purpose |
|---|---|---|---|
| NEO-6M | Position, GPS speed and time | UART | Provide positioning and movement information |
| DS18B20 | Temperature | 1-Wire | Monitor environmental temperature |
| BH1750 | Ambient light level | I²C | Record lighting conditions around the driver |

The ambient-light measurement is particularly relevant to the night-time monitoring objective because it provides information about the lighting conditions during operation.

### 📷 Driver Monitoring Camera

The vision subsystem uses a dedicated IR / night-vision camera positioned toward the driver.

```text
Driver
   ↓
IR / Night-Vision Camera
   ↓
Facial Landmark Detection
   ↓
EAR / MAR / PERCLOS / Blink Rate
   ↓
Drowsiness Analysis
---

## ⚡ Latency and Synchronization Design

Low-latency communication is an important requirement for a driver-monitoring system because sensor information from different sources needs to be processed together.

The prototype uses UDP-based communication between the ESP32-S3 sensor nodes and the Raspberry Pi 5.

### 📡 UDP Communication

Each ESP32-S3 node transmits sensor measurements to the Raspberry Pi using UDP packets.

```text
ESP32-S3 Sensor Node
        ↓
   Data Acquisition
        ↓
Sequence Number + Timestamp
        ↓
      UDP Packet
        ↓
      Wi-Fi
        ↓
Raspberry Pi 5
        ↓
Packet Validation

---

## 🧠 Drowsiness Detection Parameters

The computer-vision subsystem uses facial landmark measurements to identify behavioural indicators associated with driver drowsiness.

The current implementation combines Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), PERCLOS, blink rate, eye-state classification and yawning detection.

### 👁️ Eye Aspect Ratio (EAR)

Eye Aspect Ratio is used to estimate the degree of eye opening or closure.

The system calculates EAR independently for the left and right eyes and then uses the average value for overall eye-state classification.

```text
        Vertical Eye Distances
EAR = ───────────────────────────
        Horizontal Eye Distance
