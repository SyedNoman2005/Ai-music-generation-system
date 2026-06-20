# 🎯 Object Detection and Tracking System

An advanced **AI-powered real-time Object Detection and Multi-Object Tracking system** built using **YOLOv8, OpenCV, and Deep SORT**.
This project detects objects in video streams and assigns unique tracking IDs to follow them across frames.

---

## 🚀 Features

* 🔍 Real-time object detection using YOLOv8
* 🎯 Multi-object tracking with unique IDs (Deep SORT / SORT)
* 📦 Bounding boxes with labels and confidence scores
* 🎥 Supports webcam and video file input
* 📊 Object counting (optional)
* ⚡ Optimized for real-time performance

---

## 🧠 Technologies Used

* Python
* OpenCV
* YOLOv8 (Ultralytics)
* Deep SORT / SORT
* NumPy

---

## 📁 Project Structure

```
Object-Detection-Tracking/
│
├── models/              # YOLO model weights
├── tracker/             # Tracking algorithm (Deep SORT / SORT)
├── utils/               # Helper functions
├── main.py              # Main execution script
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SyedNoman2005/Object-Detection-Tracking.git
cd Object-Detection-Tracking
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* Linux/Mac:

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Required Versions

Make sure you are using:

* Python **3.8 – 3.11**
* OpenCV **4.x**
* Ultralytics YOLOv8 (latest)
* NumPy **>=1.21**

Example manual install:

```bash
pip install opencv-python ultralytics numpy
```

---

## ▶️ How to Run

### 🔹 Run with Webcam

```bash
python main.py --source 0
```

---

### 🔹 Run with Video File

```bash
python main.py --source video.mp4
```

---

## 🖥️ Output

* Displays live video with:

  * Bounding boxes
  * Object labels
  * Tracking IDs

---

## 📌 Example Use Cases

* Smart surveillance systems
* Traffic monitoring
* Crowd analysis
* Security applications

---

## 🚀 Future Improvements

* 🔥 Web app integration (Flask / React)
* 📊 Analytics dashboard
* 🧠 AI-based alert system
* ☁️ Cloud deployment

---

## 👨‍💻 Author

**Syed Noman**
GitHub: https://github.com/SyedNoman2005

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
