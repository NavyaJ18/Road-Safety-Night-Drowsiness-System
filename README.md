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

---

## 👁️ Eye Aspect Ratio (EAR)

Eye Aspect Ratio (EAR) is a geometric measure used to estimate the degree of eye openness from facial landmarks.

It is calculated using the vertical and horizontal distances between selected eye landmarks:

\[
EAR = \frac{||p_2-p_6|| + ||p_3-p_5||}
{2||p_1-p_4||}
\]

A higher EAR generally represents a more open eye, while a lower EAR represents reduced eye opening.

### Current Project Thresholds
---

## 😮 Mouth Aspect Ratio (MAR)

Mouth Aspect Ratio (MAR) is used to estimate the degree of mouth opening from facial landmarks.

The implementation uses multiple vertical mouth-opening measurements and normalizes them using the horizontal mouth width.

The current implementation uses three vertical mouth measurements to calculate the average mouth opening relative to mouth width.

### Current Threshold

```text
MAR >= 0.60 → Yawning Detected
MAR < 0.60  → No Yawn

```text
EAR >= 0.30        → Eyes Open
---

## ⏱️ PERCLOS

PERCLOS (Percentage of Eye Closure) represents the proportion of a defined observation period during which the driver's eyes are considered closed.

The project maintains a rolling observation window and calculates the percentage of frames classified as having closed eyes.

### Current Configuration

```text
PERCLOS Window     = 90 seconds
PERCLOS Threshold  = 80%
0.18 <= EAR < 0.30 → Partially Closed
EAR < 0.18         → Eyes Closed
---

## 👀 Blink Rate

The system monitors transitions between eye states to identify blink events.

A blink is detected when the eye state changes from open to closed and subsequently returns to open.

The detected blink events are maintained over a rolling time window.
---

## 🧠 Drowsiness Classification

The system combines multiple detected parameters to determine the current behavioural status.

The current classification logic prioritizes prolonged eye closure and then evaluates other detected conditions.

### Current Status Categories

- `NORMAL`
- `EYES CLOSED`
- `YAWNING DETECTED`
- `DROWSY`

PERCLOS is used as a longer-duration indicator, while EAR provides instantaneous eye-state information.

The classification is intended as a project-level detection mechanism and is not a medical diagnostic system.

### Current Configuration

```text
---

## 📷 Computer Vision Pipeline

The current vision pipeline follows these general steps:

1. Capture frames from the webcam.
2. Detect facial landmarks.
3. Extract relevant eye landmarks.
4. Calculate left-eye and right-eye EAR.
5. Calculate the average EAR.
6. Extract mouth landmarks.
7. Calculate MAR.
8. Determine eye state.
9. Detect blink events.
10. Calculate blink rate.
11. Maintain the PERCLOS observation window.
12. Determine the current drowsiness status.
13. Record the calculated parameters.
14. Display the results on the monitoring interface.
Blink Rate Window       = 60 seconds
Blink Rate Threshold    = 10 blinks/min
---

## 📊 Monitoring Dashboard

A Streamlit-based dashboard is used to visualize the recorded drowsiness parameters.

### Drowsiness Detection View

The dashboard provides monitoring and analysis of:

- Average EAR
- Left-eye EAR
- Right-eye EAR
- MAR
- PERCLOS
- Blink rate
- Eye status
- Yawning status
- Blink status
- Overall drowsiness status
- Total recorded observations

The dashboard also provides graphical analysis of parameter changes over time and allows recorded data to be downloaded.
