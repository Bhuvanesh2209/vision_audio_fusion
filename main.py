import os
import cv2
import csv
import openl3
import numpy as np
import soundfile as sf
from ultralytics import YOLO
from sklearn.metrics import classification_report
from collections import defaultdict

def is_poor_lighting(image_path, threshold=50):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    avg_brightness = np.mean(image)
    return avg_brightness < threshold

def detect_with_yolo(image_path):
    model = YOLO("yolov8n.pt")
    results = model(image_path)
    objects = set()
    for r in results:
        for cls_id in r.boxes.cls:
            objects.add(model.model.names[int(cls_id)])
    return list(objects)

def detect_with_openl3(audio_path):
    audio, sr = sf.read(audio_path)
    emb, _ = openl3.get_audio_embedding(audio, sr, input_repr="mel256", content_type="env", embedding_size=512)
    mean_emb = emb.mean(axis=0)
    energy = np.linalg.norm(mean_emb)
    if energy > 40:
        return ["truck", "car"]
    elif energy > 20:
        return ["person"]
    else:
        return ["ambient", "unknown"]

def load_ground_truth(gt_file):
    gt_dict = {}
    with open(gt_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row["video_id"]
            true_labels = row["true_objects"].split(",")
            gt_dict[video_id] = [label.strip() for label in true_labels]
    return gt_dict

def flatten_predictions(y_true_dict, y_pred_dict):
    all_labels = sorted({label for labels in y_true_dict.values() for label in labels} |
                        {label for labels in y_pred_dict.values() for label in labels})
    
    y_true_flat = []
    y_pred_flat = []

    for video_id in y_true_dict:
        true_labels = set(y_true_dict[video_id])
        pred_labels = set(y_pred_dict.get(video_id, []))

        for label in all_labels:
            y_true_flat.append(1 if label in true_labels else 0)
            y_pred_flat.append(1 if label in pred_labels else 0)

    return y_true_flat, y_pred_flat, all_labels

def process_video_samples(frame_dir="frames", audio_dir="audio", output_csv="predictions.csv", ground_truth_csv="ground_truth.csv"):
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    predictions = {}
    results = []

    for frame_file in frame_files:
        video_id = frame_file.replace("_frame.jpg", "")
        frame_path = os.path.join(frame_dir, frame_file)
        audio_path = os.path.join(audio_dir, f"{video_id}.wav")

        if not os.path.exists(audio_path):
            print(f"⚠️  Missing audio for {video_id}, skipping...")
            continue

        if is_poor_lighting(frame_path):
            print(f"🔇 Poor lighting in {frame_file} -> using audio")
            labels = detect_with_openl3(audio_path)
        else:
            print(f"🖼️  Good lighting in {frame_file} -> using YOLOv8")
            labels = detect_with_yolo(frame_path)

        print(f"✅ {video_id} -> {labels}")
        predictions[video_id] = labels
        results.append((video_id, ", ".join(labels)))

    # Save predictions to CSV
    with open(output_csv, "w") as f:
        f.write("video_id,predicted_objects\n")
        for video_id, objs in results:
            f.write(f"{video_id},{objs}\n")
    print(f"\n📁 Results saved to {output_csv}")

    # --- Evaluation ---
    if os.path.exists(ground_truth_csv):
        gt_labels = load_ground_truth(ground_truth_csv)
        y_true, y_pred, label_names = flatten_predictions(gt_labels, predictions)

        print("\n📊 Evaluation Metrics:")
        print(classification_report(y_true, y_pred, target_names=["not_" + l for l in label_names] + label_names))
    else:
        print("⚠️ Ground truth file not found. Skipping evaluation.")

if __name__ == "__main__":
    process_video_samples()
