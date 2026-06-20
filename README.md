# 🎵 AI Music Generation System

An advanced **AI-powered music generation system** that creates original musical compositions using **Deep Learning (LSTM)** trained on MIDI datasets.
This project learns patterns in music and generates new melodies with controllable creativity.

---

## 🚀 Features

* 🎼 Generate original music using AI
* 🧠 LSTM-based deep learning model
* 🎹 MIDI file processing using `music21`
* 🎛️ Control creativity using temperature sampling
* 💾 Save generated music as MIDI files
* 🔊 Optional audio playback support
* ⚡ Modular and scalable architecture

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* music21
* NumPy
* MIDI datasets

---

## 📁 Project Structure

```bash
Ai-music-generation-system/
│
├── dataset/                # MIDI files dataset
├── model/                  # Saved trained models
├── output/                 # Generated music files
├── data_processing.py      # MIDI preprocessing
├── model.py                # LSTM model architecture
├── train.py                # Model training script
├── generate.py             # Music generation script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SyedNoman2005/Ai-music-generation-system.git
cd Ai-music-generation-system
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
* TensorFlow **2.x**
* music21 **>=7.0**
* NumPy **>=1.21**

Example manual install:

```bash
pip install tensorflow music21 numpy
```

---

## 🧪 How It Works

1. 📥 Load MIDI dataset
2. 🎹 Extract notes and chords using `music21`
3. 🔢 Convert into numerical sequences
4. 🧠 Train LSTM model on sequences
5. 🎶 Generate new note sequences
6. 💾 Convert output back to MIDI

---

## ▶️ How to Run

### 🔹 Step 1: Train the Model

```bash
python train.py
```

---

### 🔹 Step 2: Generate Music

```bash
python generate.py
```

---

## 🎧 Output

* Generated music is saved in the `output/` folder
* Format: `.mid` (MIDI file)
* Can be played using any MIDI player or DAW

---

## 📌 Example Use Cases

* AI-based music composition
* Game background music generation
* Creative AI experiments
* Music research projects

---

## 🚀 Future Improvements

* 🌐 Web app interface (React + FastAPI)
* 🎼 Multi-instrument music generation
* 🎛️ Style-based music generation (classical, jazz)
* 🔥 Transformer-based models (Music Transformer)
* 🎹 Piano roll visualization

---

## 👨‍💻 Author

**Syed Noman**
GitHub: https://github.com/SyedNoman2005

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
