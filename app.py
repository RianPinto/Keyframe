import streamlit as st
import cv2
import torch
import numpy as np
import ruptures as rpt
import hdbscan

from PIL import Image
from sklearn.decomposition import PCA
from torchvision import models, transforms
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

st.title("Video Visual + Text Summary")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_models():

    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval().to(device)

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    llm = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large"
    ).to(device)

    return resnet, processor, caption_model, tokenizer, llm


resnet, processor, caption_model, tokenizer, llm = load_models()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def sample_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps/2)

    frames = []
    timestamps = []

    i = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if i % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            timestamps.append(i/fps)

        i += 1

    cap.release()

    return frames, timestamps


def extract_features(frames):

    feats = []

    with torch.no_grad():

        for frame in frames:

            img = Image.fromarray(frame)
            x = transform(img).unsqueeze(0).to(device)

            f = resnet(x)
            f = f.squeeze().cpu().numpy()

            feats.append(f)

    return np.array(feats)


def keyframe_selection(frames, features):

    diffs = np.linalg.norm(features[1:] - features[:-1], axis=1)

    algo = rpt.Pelt(model="rbf").fit(diffs)
    algo.predict(pen=10)

    pca = PCA(n_components=100)
    reduced = pca.fit_transform(features)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(reduced)

    selected = []

    for c in set(labels):

        if c == -1:
            continue

        idx = np.where(labels == c)[0]

        cluster_feats = reduced[idx]

        centroid = cluster_feats.mean(axis=0)

        d = np.linalg.norm(cluster_feats-centroid,axis=1)

        best = idx[np.argmin(d)]

        selected.append(best)

    return sorted(selected)


def caption_frames(frames, indices):

    captions = []

    for i in indices:

        img = Image.fromarray(frames[i])

        inputs = processor(img, return_tensors="pt").to(device)

        out = caption_model.generate(**inputs,max_length=30)

        caption = processor.decode(out[0], skip_special_tokens=True)

        captions.append(caption)

    return captions


def summarize(captions):

    prompt = f"""
These sentences describe keyframes from a video:

{captions}

Write a coherent paragraph summarizing the video.
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    out = llm.generate(**inputs,max_length=200)

    summary = tokenizer.decode(out[0],skip_special_tokens=True)

    return summary


uploaded = st.file_uploader("Upload Video", type=["mp4","mov","avi"])

if uploaded:

    with open("temp_video.mp4","wb") as f:
        f.write(uploaded.read())

    st.video("temp_video.mp4")

    if st.button("Generate Summary"):

        with st.spinner("Sampling frames..."):
            frames, ts = sample_frames("temp_video.mp4")

        with st.spinner("Extracting features..."):
            features = extract_features(frames)

        with st.spinner("Selecting keyframes..."):
            selected = keyframe_selection(frames, features)

        st.subheader("Keyframes")

        cols = st.columns(3)

        for i, idx in enumerate(selected):
            with cols[i % 3]:
                st.image(frames[idx], use_container_width=True)

        with st.spinner("Generating captions..."):
            captions = caption_frames(frames, selected)

        with st.spinner("Generating final summary..."):
            summary = summarize(captions)

        st.subheader("Video Summary")

        st.write(summary)