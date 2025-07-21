# Vision-Audio Fusion System 🎥🔊

A dual-modality object detection system that intelligently switches between **YOLOv8 (vision)** and **OpenL3 (audio)** models to identify objects in video clips based on lighting conditions.

> ⚡️ In good lighting → Vision-based detection using YOLOv8  
> 🌙 In poor lighting → Audio-based classification using OpenL3 embeddings

---

## 📌 Project Overview

This project addresses real-world limitations in object detection under poor visual conditions (e.g., night scenes, fog, motion blur) by leveraging audio cues when vision fails.

It performs:
- Object detection from video frames using YOLOv8
- Audio classification using OpenL3 embeddings
- Lighting analysis to choose the optimal modality
- CSV output of all predictions

---

## 🧠 Architecture

```text
           ┌───────────────┐
           │   Frame.jpg   │
           └─────┬─────────┘
                 │
       Bright? ──┼────────────┐
        (cv2)    │            │
          Yes    ↓            ↓
        ┌──────────────┐  ┌────────────────┐
        │ Use YOLOv8   │  │ Use OpenL3     │
        └──────────────┘  └────────────────┘
                 │             │
            Detected        Embeddings
             Objects           │
                 └────┬────────┘
                      ↓
            Save predictions
📁 Directory Structure
bash
Copy
Edit
vision_audio_fusion/
│
├── frames/                # Extracted video frames (JPG)
├── audio/                 # Extracted audio clips (WAV)
├── main.py                # Run detection pipeline
├── extract_data.py        # Feature extraction logic
└── predictions.csv        # Output predictions
⚙️ Requirements
Install dependencies inside a virtual environment:

bash
Copy
Edit
python -m venv cv
source cv/bin/activate
pip install -r requirements.txt
Minimal requirements.txt:

txt
Copy
Edit
ultralytics
openl3
librosa
soundfile
tensorflow
numpy
opencv-python
🚀 Usage
1. Extract frames and audio (if not already):
bash
Copy
Edit
# Add videos and extract frames + audio using your own script/tool
2. Train the model
bash
Copy
Edit
python train.py
3. Predict with a new sample
bash
Copy
Edit
python predict.py frames/sample_frame.jpg audio/sample.wav
4. Run entire pipeline on dataset
bash
Copy
Edit
python main.py
🧪 Example Output
rust
Copy
Edit
Good lighting in sample1.jpg -> using YOLOv8
sample1 -> ['car', 'person']

Poor lighting in sample2.jpg -> using audio
sample2 -> ['truck', 'ambient']
