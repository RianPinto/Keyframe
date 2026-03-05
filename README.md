# Keyframe Enrichment & Text Summary

This project generates a visual and textual summary of a video. 

The application extracts representative keyframes from a video and produces a coherent textual description of the video content using deep learning models. A Streamlit web interface allows users to upload a video and automatically obtain a summary.

<img width="975" height="748" alt="image" src="https://github.com/user-attachments/assets/c5b5169c-afa3-42cf-88ae-a2829d5de4ca" />
<img width="975" height="782" alt="image" src="https://github.com/user-attachments/assets/c02ff693-a575-417b-b58f-b01b1a6d5a29" />
<img width="975" height="631" alt="image" src="https://github.com/user-attachments/assets/45341db8-4aac-480b-805c-6932c8cae7d8" />


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
git clone https://github.com/RianPinto/Keyframe.git
pip install -r requirements.txt
streamlit run app.py
```
