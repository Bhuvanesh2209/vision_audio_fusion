import os
import librosa
import soundfile as sf
from PIL import Image
import openl3
from ultralytics import YOLO

def extract_audio_features(audio_path):
    audio, sr = sf.read(audio_path)
    emb, _ = openl3.get_audio_embedding(audio, sr, input_repr="mel256", content_type="env", embedding_size=512)
    return emb.mean(axis=0)  # average pooling

def extract_visual_features(image_path):
    model = YOLO("yolov8n.pt")
    results = model(image_path)
    detected = []
    for r in results:
        for cls_id in r.boxes.cls:
            detected.append(model.model.names[int(cls_id)])
    return list(set(detected))  # unique detected classes