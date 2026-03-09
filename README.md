# BiMER – Bimodal Emotion Recognition System

A bimodal emotion recognition framework that analyzes **speech and text** to predict human emotions using modern transformer-based models.

This project implements a multimodal emotion recognition system combining **Wav2Vec 2.0 for speech emotion recognition** and **RoBERTa for text emotion recognition**, with a **late fusion strategy** to produce the final prediction.

The system is designed to operate under three scenarios:

* Speech-only emotion recognition
* Text-only emotion recognition
* Bimodal emotion recognition (speech + text)

By combining complementary cues from acoustic signals and linguistic content, the system improves overall emotion classification performance.

---

# Live Demo

Try the system online:
huggingface.co/spaces/mrflex/BIMER-Emotion-Recognition
(Note: Wait for the webpage to load. It may take some time)

---

# Project Overview

Emotion recognition plays a critical role in **affective computing** and **human–computer interaction**. Humans express emotions through multiple modalities such as speech tone, facial expressions, and textual language. Relying on a single modality often leads to incomplete interpretation of emotional states.

This project introduces a **bimodal emotion recognition system** that processes speech and text independently and combines their predictions at the decision stage using a **late fusion strategy**.

The system leverages transformer-based models to capture emotional cues from both modalities:

* **Speech Emotion Recognition (SER)** using Wav2Vec 2.0
* **Text Emotion Recognition (TER)** using RoBERTa

The final emotion prediction is obtained by integrating the outputs of both models.
Experimental evaluation on the **IEMOCAP dataset** shows that combining speech and text improves overall classification accuracy compared to unimodal approaches. 

---

# Key Features

* Bimodal emotion recognition using **speech and text**
* Transformer-based models for both modalities
* Modular architecture allowing **speech-only, text-only, or combined inference**
* Late fusion strategy for prediction integration
* Evaluation on the **IEMOCAP emotion dataset**
* Implementation using **PyTorch and Hugging Face Transformers**

---

# System Architecture

The framework consists of two independent processing branches:

### Speech Emotion Recognition (SER)

* Input: Raw speech waveform
* Encoder: **Wav2Vec 2.0**
* Feature extraction from raw audio
* Classification head predicts emotion probabilities

### Text Emotion Recognition (TER)

* Input: Text transcripts
* Encoder: **RoBERTa**
* Contextual representation learning
* Classification layer produces emotion prediction scores

### Fusion Mechanism

The outputs of the two models are combined using a **late fusion strategy at the logit level**.
This approach preserves modularity while allowing both modalities to contribute to the final prediction.

The architecture processes speech and text independently before integrating their prediction scores to determine the final emotion label. 

---

# Emotion Classes

The system predicts four emotion categories:

* Angry
* Happy
* Sad
* Neutral

These categories are commonly used in emotion recognition research based on the IEMOCAP dataset.

---

# Dataset

The system is evaluated using the **Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset**, a widely used benchmark for multimodal emotion recognition research.

The dataset contains conversational recordings from professional actors with annotated emotional labels and corresponding text transcripts. 

### Dataset Statistics

| Emotion | Utterances |
| ------- | ---------- |
| Angry   | 1147       |
| Happy   | 1807       |
| Sad     | 1110       |
| Neutral | 1790       |
| Total   | 5854       |

The dataset is split using a **speaker-independent setup**:

* Sessions 1–4 → Training
* Session 5 → Testing

This ensures the model generalizes to unseen speakers during evaluation. 

---

# Model Configuration

| Component        | Model                  |
| ---------------- | ---------------------- |
| Speech Encoder   | Wav2Vec 2.0            |
| Text Encoder     | RoBERTa-base           |
| Fusion Strategy  | Late Fusion            |
| Loss Function    | Weighted Cross Entropy |
| Speech Optimizer | Adam                   |
| Text Optimizer   | AdamW                  |

Training is performed independently for speech and text models, while fusion is applied during inference.

---

# Experimental Results

| Model                   | Accuracy  |
| ----------------------- | --------- |
| Speech-only             | 54.8%     |
| Text-only               | 68.9%     |
| Bimodal (Speech + Text) | **72.0%** |

The results demonstrate that combining speech and text improves classification accuracy compared to using either modality alone. 

---

# Repository Structure

```
bimer-emotion-recognition

notebooks/
    bimer_final.ipynb        Training and evaluation notebook

saved_models/
    ter_tokenizer/           Tokenizer files for text model

Docs/images/
    Project diagrams and figures

requirements.txt
.gitignore
README.md
```

Large datasets and environments are intentionally excluded from the repository.

---

# Installation

Clone the repository:

```
git clone https://github.com/Vishnu539/bimer-emotion-recognition.git
cd bimer-emotion-recognition
```

Create a virtual environment:

```
python -m venv venv
```

Activate the environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

notebooks/bimer_final.ipynb
```

Or run the interactive web app:

```bash
python app.py
```

The notebook contains:

* Data preprocessing
* Model training
* Evaluation
* Bimodal fusion implementation

---

# Important Notes

The following files are **not included in the repository**:

* IEMOCAP dataset
* Large trained model weights
* Virtual environment files
* Documents and reports

This is standard practice for machine learning repositories to keep the repository lightweight.

---

# How to Obtain the Dataset

Download the dataset from:

https://sail.usc.edu/iemocap/

After downloading, place the dataset inside your project directory before running the notebook.

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Wav2Vec 2.0
* RoBERTa
* NumPy
* Pandas
* Scikit-learn
* Jupyter Notebook

---

# Applications

Emotion recognition systems have applications in many domains:

* Human–computer interaction
* Mental health monitoring
* Customer service analytics
* Conversational AI systems
* Smart assistants

---

# Future Work

Possible extensions of this system include:

* Real-time emotion recognition
* Dynamic fusion strategies
* Improved speech model fine-tuning
* Integration of facial emotion recognition
* Evaluation on larger multimodal datasets

---

# Authors

Guvvala Trinisha
Chilakala Vishnuvardhan Reddy
Bellamkonda Kumara Swamy
Madamala Manivardhan
Kurma Laya Sri

Department of Computer Science and Engineering (AI & ML)
S R Gudlavalleru Engineering College

---

# References

1. Dikbiyik et al., **BiMER: Design and Implementation of a Bimodal Emotion Recognition System**, IEEE Access.
2. Busso et al., **IEMOCAP: Interactive Emotional Dyadic Motion Capture Database**.
3. Baevski et al., **Wav2Vec 2.0: A Framework for Self-Supervised Speech Representation Learning**.
4. Liu et al., **RoBERTa: A Robustly Optimized BERT Pretraining Approach**.
