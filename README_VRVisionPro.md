# XR Teleoperate with VRVisionPro Display

<div align="center">
  <h1>xr_teleoperate</h1>
  <p>
    <strong>Next-Gen Humanoid Teleoperation System with Advanced VR Visualization</strong>
  </p>
  <p align="center">
    <a> English </a> | <a href="README_zh-CN.md">中文</a> | <a href="README_ja-JP.md">日本語</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Platform-Apple%20Vision%20Pro%20%7C%20Quest%203%20%7C%20PICO%204-blue" alt="Platform">
    <img src="https://img.shields.io/badge/Robot-Unitree%20G1%20%7C%20H1-orange" alt="Robot">
    <img src="https://img.shields.io/badge/Vision-VRVisionPro%20HUD-green" alt="Vision">
  </p>
</div>

---

## 🚀 Overview

**xr_teleoperate** is a comprehensive teleoperation framework designed for **Unitree Humanoid Robots** (G1, H1). It leverages cutting-edge **XR (Extended Reality)** devices to provide an intuitive, low-latency control interface.

At the core of the user experience is the **VRVisionPro Display System**—a sophisticated frontend visualization engine that enhances teleoperation with intelligent AR overlays, real-time object segmentation, and task-specific guidance.

## 👓 VRVisionPro Frontend Display

The **VRVisionPro** module (integrated via `televuer` and `teleimager`) redefines how operators perceive the remote environment. It transforms raw camera feeds into an augmented, information-rich dashboard.

### 🌟 Key Features

#### 1. Multi-Mode Visualization
Adapt the visual interface to your operational needs:
- **🔮 Immersive Mode**: Full-screen First-Person View (FPV) from the robot's head camera. Complete immersion for precision tasks.
- **🖼️ Ego Mode**: Picture-in-Picture (PiP) display. The robot's view is centered, surrounded by the headset's passthrough view for local situational awareness.
- **👓 Pass-through Mode**: Augmented Reality (AR) overlay. View the real world through the headset cameras with robot telemetry overlay (when applicable).

#### 2. Intelligent HUD & Task Guidance
The system features a futuristic, Sci-Fi inspired Heads-Up Display (HUD) powered by real-time computer vision (YOLO/Segmentation):

*   **Task 1: Arrange Items**
    *   **Workflow**: Intelligent detection of **Bottles** (Active) and **Books** (Anchor).
    *   **Guidance**: Dynamic yellow dashed lines visualize the optimal path from Object to Target.
    *   **Feedback**: Green perspective targets appear automatically relative to the anchor object.
    
*   **Task 3: Human Safety / Crisis Response**
    *   **Risk Detection**: **Bottles** are flagged as "RISK OBJECTS" (Red/Orange alert).
    *   **Safe Zones**: **Cups** act as "ANCHOR" points for safe placement.
    *   **Visual Safety**: Clear textual warnings and distinct color-coding (Red for danger, Blue for neutral).

*   **AR Segmentation**:
    *   Objects are highlighted with translucent fills (Yellow/Cyan) and bounding contours.
    *   Real-time status counters ("X TARGETS FOUND").

#### 3. Low-Latency Streaming
*   **WebRTC / ZMQ**: Optimized for real-time video transmission over Wi-Fi.
*   **Cross-Platform**: Compatible with **Apple Vision Pro**, **Meta Quest 3**, and **PICO 4 Ultra Enterprise**.

---

## 📦 Video Demo

<p align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="https://img.youtube.com/vi/OTWHXTu09wE/maxresdefault.jpg" alt="G1 Teleop" width="90%">
        <p><b> G1 (29DoF) Teleoperation </b></p>
      </td>
      <td align="center" width="50%">
        <img src="https://img.youtube.com/vi/pNjr2f_XHoo/maxresdefault.jpg" alt="H1 Teleop" width="90%">
        <p><b> H1 (Arm 7DoF) Teleoperation </b></p>
      </td>
    </tr>
  </table>
</p>

---

## 🛠️ System Architecture

1.  **Robot Backend**:
    *   `teleop/robot_control`: Handles Inverse Kinematics (IK) and motion retargeting for G1/H1 arms and hands (Dex3/Inspire/BrainCo).
2.  **Vision Server (`teleimager`)**:
    *   Captures high-res video from robot cameras.
    *   Distributes config (camera intrinsics/extrinsics) to clients.
3.  **Frontend Client (`televuer` / VRVisionPro)**:
    *   Receives video via WebRTC/ZMQ.
    *   Renders the **HUDRenderer** layer (Overlays, AR tags).
    *   Handles VR input (Hand tracking, Head movement/IK).

## 🏃 Quick Start

### Prerequisites
*   Ubuntu 20.04/22.04
*   Python 3.8+
*   Supported XR Headset (AVP, Quest3, PICO4)

### Installation

```bash
git clone https://github.com/unitreerobotics/xr_teleoperate.git
cd xr_teleoperate
pip install -r requirements.txt
```

### Running the System

1.  **Start the Teleop Node** (on the PC controlling the robot):
    ```bash
    cd teleop
    python teleop_hand_and_arm.py --input-mode=controller --arm=G1_29 --ee=dex3
    ```

2.  **Connect VR Headset**:
    *   Ensure the headset and PC are on the same local network.
    *   Open the Vuer web interface (link provided in terminal) in the headset's browser.
    *   Click **"ENTER VR"**.

3.  **Select Task Visualization**:
    *   In the terminal interface, select a Task ID (e.g., `1` for Arrange, `3` for Safety) to enable specific VRVisionPro HUD overlays.

---

## 📝 License
This project is licensed under the BSD-3-Clause License.
