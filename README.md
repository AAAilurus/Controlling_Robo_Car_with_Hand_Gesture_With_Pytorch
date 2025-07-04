# Controlling_Robo_Car_with_Hand_Gesture_With_Pytorch
# Hand Gesture Controlled Robot Car (PyTorch + OpenCV + Flask + Raspberry Pi + Arduino + Bluetooth)

This project demonstrates a hand gesture-controlled robot car using a laptop (with webcam), Raspberry Pi, Arduino Uno, HC-05 Bluetooth module, and L298N motor driver. The system uses a custom-trained PyTorch model to detect hand gestures in real-time. The gesture commands are sent from the laptop to the Raspberry Pi over HTTP (using Flask), and from the Pi to the Arduino via Bluetooth, which then drives the motors accordingly.

---

## Table of Contents

- Overview
- System Architecture
- Hardware Requirements
- Software Requirements
- Hand Gestures and Commands
- Project Structure
- Setup Instructions
- Communication Flow
- Example Code Snippets
- Roadmap
- Author
- License

---

## Overview

Due to limited support for MediaPipe on Raspberry Pi, this system offloads gesture recognition to a more powerful laptop. The laptop uses OpenCV and PyTorch to recognize hand gestures from a webcam. Recognized gestures are sent to the Raspberry Pi via HTTP using a Flask server. The Pi then transmits the command to the Arduino over Bluetooth, which drives the motors using an L298N motor driver module.

---

## System Architecture

Laptop (PyTorch + OpenCV + Flask Client)  
→ HTTP POST →  
Raspberry Pi (Flask Server + Bluetooth TX)  
→ Bluetooth →  
Arduino Uno (L298N Motor Driver + Motors)

---

## Hardware Requirements

- Laptop with Python and webcam (built-in or USB)
- Raspberry Pi (3/4 recommended with Python and Bluetooth)
- Arduino Uno
- HC-05 Bluetooth module
- L298N motor driver
- 2 DC motors with wheels
- Chassis and caster wheel
- Battery pack (6V–12V for motors)
- Jumper wires and breadboard (if needed)

---

## Software Requirements

### On Laptop:
- Python 3.8+
- OpenCV
- PyTorch
- Flask (as HTTP client)
- Webcam (for gesture detection)

### On Raspberry Pi:
- Python 3+
- Flask (as HTTP server)
- PySerial (for sending serial data to Bluetooth)

### On Arduino:
- Arduino IDE
- Bluetooth Serial command receiver sketch

---

## Hand Gestures and Corresponding Commands

| Gesture        | Action     | Command |
|----------------|------------|---------|
| Open Palm      | Stop       | S       |
| Fist           | Forward    | F       |
| Point Left     | Left       | L       |
| Point Right    | Right      | R       |
| Thumb Down     | Backward   | B       |

---

