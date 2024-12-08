from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, pipeline
from PIL import Image
import torch
from scipy.fftpack import dct
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

import base64
from io import BytesIO

@app.route('/video', methods=['POST'])
def process_video():
    video_file = request.files.get('video')
    if not video_file:
        return jsonify({"error": "No video file uploaded."}), 400

    video_path = './uploaded_video.mp4'
    video_file.save(video_path)

    # Step 2: Extract frames
    print("Extract Frames ")
    frames_dir = './frames/'
    os.makedirs(frames_dir, exist_ok=True)
    cam = cv2.VideoCapture(video_path)
    total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frames):
        ret, frame = cam.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{i}.jpg"), frame)
    cam.release()

    print("Calculate intensity differences")
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    intensity_differences = []
    for i in range(len(frame_files) - 1):
        frame1 = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame_files[i + 1], cv2.IMREAD_GRAYSCALE)
        diff = cv2.absdiff(frame1, frame2)
        intensity_differences.append(np.sum(diff))

    # Step 4: Select key frames based on intensity differences
    print("Select Key frames based on intensity differences")
    threshold = np.percentile(intensity_differences, 95)  # Top 5% intensity differences
    key_frames = [frame_files[i] for i, diff in enumerate(intensity_differences) if diff >= threshold]

    print("Cluster Key frames")
    # Step 5: Cluster key frames
    feature_vectors = [cv2.imread(f, cv2.IMREAD_GRAYSCALE).flatten() for f in key_frames]
    scaler = StandardScaler()
    feature_vectors_standardized = scaler.fit_transform(feature_vectors)
    dct_vectors = np.apply_along_axis(lambda x: dct(x, norm='ortho'), 1, feature_vectors_standardized)
    optics = OPTICS()
    clusters = optics.fit_predict(dct_vectors)

    # Select one representative frame per cluster
    print("Select representative keyframe")
    cluster_representatives = []
    for cluster in set(clusters):
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster]
        if cluster_indices:
            cluster_representatives.append(key_frames[cluster_indices[len(cluster_indices) // 2]])

    # Step 6: Generate captions for key frames
    print("Generating caption per keyframe")
    captions = []
    encoded_images = []  # List to store Base64-encoded images
    for frame_path in cluster_representatives:
        image = Image.open(frame_path).convert("RGB")

        # Encode the image to Base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_images.append(encoded_image)

        # Generate caption
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        output_ids = captioning_model.generate(pixel_values, max_length=16, num_beams=4)
        captions.append(tokenizer.decode(output_ids[0], skip_special_tokens=True).strip())

    # Step 7: Summarize captions
    print("Summarizing caption")
    global_captions = " ".join(captions)
    summary = summarizer(global_captions, max_length=100, min_length=0, do_sample=False)[0]['summary_text']

    # Cleanup
    os.remove(video_path)
    for f in frame_files:
        os.remove(f)
    os.rmdir(frames_dir)

    # Return the summary and keyframes as Base64-encoded strings
    return jsonify({
        "summary": summary,
        "keyframes": encoded_images
    })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
