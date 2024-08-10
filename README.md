# TaipeiTech_FishRobot_Control

This repository contains the code and resources for the **Hand Gesture and Voice Control of a Bionic Robotic Fish**, developed as part of an AI Robotics project at Taipei Tech. The project aims to provide an intuitive and efficient way to control a robotic fish using hand gestures and voice commands, leveraging the lightweight AI capabilities optimized for NVIDIA's Jetson Nano platform.

---

## Project Description

The **TaipeiTech_FishRobot_Control** project showcases the integration of AI for real-time control of a bionic robotic fish. This control system includes:

- **Hand Gesture Recognition**: A model capable of recognizing and interpreting various hand gestures, allowing users to control the fish’s movements in water.
- **Voice Command Integration**: A voice control system that enables hands-free operation by recognizing specific voice commands.
- **Optimization for Jetson Nano**: The AI models have been meticulously optimized to run on Jetson Nano, ensuring smooth performance even with low frames per second (FPS).

### Context

This project was developed as part of an advanced AI Robotics course at Taipei Tech. The primary focus was to create a responsive and interactive control mechanism for a robotic fish that can be deployed in educational, research, or hobbyist environments. The use of Jetson Nano as the computational platform highlights the model’s efficiency and suitability for edge computing scenarios.

---

## Getting Started

### Prerequisites

Before running the code, ensure your environment meets the following requirements:

- **Hardware**: [NVIDIA Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano)
- **Software**: Python 3.6 or higher installed
- **Dependencies**: Install the required Python libraries with the following command:

```bash
pip install -r requirements.txt

Running the Hand Gesture Recognition Model
To start the hand gesture recognition system:


```bash
python hands_gesture_recog.py
This script initiates the hand gesture recognition model, enabling real-time control of the robotic fish using predefined gesture
