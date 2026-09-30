# 🚗 Driver Drowsiness Detection and Road Safety Monitoring System

A real time computer vision-based system designed to monitor driver alertness and identify potential signs of drowsiness. The project analyzes facial features through a live camera feed and calculates multiple parameters associated with driver fatigue.

## 📌 Project Overview

Driver drowsiness is a significant road safety concern, particularly during long distance and night time driving. This project aims to detect behavioural indicators of fatigue using facial landmark analysis.

The system currently monitors:

- 👁️ Eye Aspect Ratio (EAR)
- 😮 Mouth Aspect Ratio (MAR)
- ⏱️ PERCLOS
- 👀 Eye state classification
- 🥱 Yawning detection
- 📊 Timestamp-based data logging
- 📈 Data visualization through a dashboard

The system is designed to be further extended for nighttime road-safety experiments and additional posture analysis.

---

## 🔍 Parameters Used

### 1. Eye Aspect Ratio (EAR)

EAR measures the openness of the eyes using facial landmarks.

\[
EAR = \frac{||p_2-p_6|| + ||p_3-p_5||}
{2||p_1-p_4||}
\]

Current threshold:

```text
EAR Threshold = 0.30

---

## 🔧 Current System Components

The current implementation includes:

- Webcam-based facial landmark detection
- Eye Aspect Ratio (EAR) calculation
- Mouth Aspect Ratio (MAR) calculation
- PERCLOS-based eye-closure analysis
- Blink rate monitoring
- Eye state classification
- Yawning detection
- Timestamp-based data logging
- Streamlit dashboard for parameter monitoring and visualization
- UDP-based sensor communication
- Multi-sensor data simulation
- Sensor data logging with sequence numbers and latency tracking
