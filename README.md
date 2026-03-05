# Video Visual & Text Summary

This project generates a visual and textual summary of a video. 

The application extracts representative keyframes from a video and produces a coherent textual description of the video content using deep learning models. A Streamlit web interface allows users to upload a video and automatically obtain a summary.

## Features

* **Upload any video** through a web interface
* **Automatic keyframe extraction**
* **Deep feature extraction** using ResNet50
* **Frame clustering** using HDBSCAN
* **Image caption generation** using BLIP
* **Final textual summary** using FLAN-T5
* **Interactive frontend** built with Streamlit

## Project Pipeline

The summarization process follows this pipeline:

```text
Video
  ↓
Frame Sampling
  ↓
Deep Feature Extraction (ResNet50)
  ↓
Dimensionality Reduction (PCA)
  ↓
Frame Clustering (HDBSCAN)
  ↓
Keyframe Selection
  ↓
Image Caption Generation (BLIP)
  ↓
Text Summarization (FLAN-T5)
  ↓
Final Video Summary
```

# To run

```text
clone this repo 
pip install -r requirements.txt
streamlit run app.py
```
