# 🚗 Driver Drowsiness Detection and Road Safety Monitoring System

A real-time computer vision-based system designed to monitor driver alertness and identify potential signs of drowsiness.

The system analyzes facial features through a live camera feed and calculates multiple behavioural parameters associated with driver fatigue. The project also includes a monitoring dashboard for visualization, data logging, and analysis.

## 📌 Project Overview

Driver drowsiness is an important road-safety concern, particularly during long-distance and night-time driving. Reduced alertness can affect reaction time, attention, and driving performance.

This project focuses on detecting behavioural indicators of drowsiness using facial landmark analysis and continuously monitoring relevant parameters.

## 🎯 Project Objectives

- Detect potential signs of driver drowsiness using computer vision.
- Monitor eye and mouth-related facial parameters.
- Analyze prolonged eye closure using PERCLOS.
- Monitor blink behaviour.
- Detect yawning behaviour.
- Record detected parameters with timestamps.
- Visualize recorded data through an interactive dashboard.
- Develop a foundation for integration with additional sensors.
- Explore network-based sensor data acquisition using UDP communication.

---

## 🔍 Parameters Used

The current system uses multiple behavioural parameters to analyze driver alertness:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- PERCLOS
- Eye state
- Blink rate
- Yawning status
- Overall drowsiness status
