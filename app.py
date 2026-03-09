import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import gradio as gr
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Model

# ----------------------------
# Device configuration
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Model Definitions
# ----------------------------

class TERModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]
        cls_repr = self.dropout(cls_repr)
        return self.classifier(cls_repr)

class SERClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Model Downloading Logic (for HF Spaces)
# ----------------------------

MODELS = {
    "ter_model.pt": "https://huggingface.co/spaces/Vishnu539/emotion-recognition/resolve/main/saved_models/ter_model.pt",
    "ser_classifier.pt": "https://huggingface.co/spaces/Vishnu539/emotion-recognition/resolve/main/saved_models/ser_classifier.pt"
}

def download_weights():
    import requests
    os.makedirs("saved_models", exist_ok=True)
    for filename, url in MODELS.items():
        path = os.path.join("saved_models", filename)
        if not os.path.exists(path):
            print(f"Downloading {filename} from Hugging Face...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")

# Call download before loading
download_weights()

# ----------------------------
# Load Models
# ----------------------------

print("Loading models...")

# Emotion mapping
ID2LABEL = {
    0: "Angry",
    1: "Sad",
    2: "Happy",
    3: "Neutral"
}

# 1. Text Emotion Recognition (TER)
ter_model_path = os.path.join("saved_models", "ter_model.pt")
ter_tokenizer_path = os.path.join("saved_models", "ter_tokenizer")

ter_tokenizer = RobertaTokenizer.from_pretrained("roberta-base") # Default if local fails
if os.path.exists(ter_tokenizer_path):
    try:
        ter_tokenizer = RobertaTokenizer.from_pretrained(ter_tokenizer_path)
    except Exception as e:
        print(f"Warning: Could not load local tokenizer: {e}")

ter_model = TERModel().to(device)
if os.path.exists(ter_model_path):
    ter_model.load_state_dict(torch.load(ter_model_path, map_location=device))
    print("✅ TER model weights loaded.")
else:
    print("⚠️ TER model weights not found. Running with uninitialized weights.")

ter_model.eval()

# 2. Speech Emotion Recognition (SER)
ser_classifier_path = os.path.join("saved_models", "ser_classifier.pt")

wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
wav2vec.eval()

ser_classifier = SERClassifier().to(device)
if os.path.exists(ser_classifier_path):
    ser_classifier.load_state_dict(torch.load(ser_classifier_path, map_location=device))
    print("✅ SER classifier weights loaded.")
else:
    print("⚠️ SER classifier weights not found. Running with uninitialized weights.")

ser_classifier.eval()

# ----------------------------
# Prediction Functions
# ----------------------------

def predict_text(text):
    if not text or not text.strip():
        return "Please enter some text."
    
    inputs = ter_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = ter_model(inputs["input_ids"], inputs["attention_mask"])
        probs = F.softmax(logits, dim=1)[0]

    pred = torch.argmax(probs).item()
    
    res = f"📝 **Text Emotion Recognition (TER)**\n\n"
    res += f"**Predicted Emotion: {ID2LABEL[pred]}**\n"
    res += f"Confidence: {probs[pred]*100:.2f}%\n\n"
    res += "Class Probabilities:\n"
    for i, label in ID2LABEL.items():
        res += f"- {label}: {probs[i]*100:.2f}%\n"
    return res

def predict_audio(audio_path):
    if not audio_path:
        return "Please upload an audio file."
    
    try:
        waveform, sr = sf.read(audio_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)

        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        max_len = 16000 * 6
        if waveform.shape[0] > max_len:
            waveform = waveform[:max_len]
        else:
            waveform = F.pad(waveform, (0, max_len - waveform.shape[0]))

        waveform = waveform.unsqueeze(0).to(device)

        with torch.no_grad():
            features = wav2vec(waveform).last_hidden_state.mean(dim=1)
            logits = ser_classifier(features)
            probs = F.softmax(logits, dim=1)[0]

        pred = torch.argmax(probs).item()
        
        res = f"🎤 **Speech Emotion Recognition (SER)**\n\n"
        res += f"**Predicted Emotion: {ID2LABEL[pred]}**\n"
        res += f"Confidence: {probs[pred]*100:.2f}%\n\n"
        res += "Class Probabilities:\n"
        for i, label in ID2LABEL.items():
            res += f"- {label}: {probs[i]*100:.2f}%\n"
        return res
    except Exception as e:
        return f"❌ Error processing audio: {e}"

def predict_bimodal(text, audio):
    results = []
    if text and text.strip():
        results.append(predict_text(text))
    if audio:
        results.append(predict_audio(audio))
    
    if not results:
        return "❌ Please enter text OR upload an audio file."
    
    return "\n" + "="*30 + "\n".join(results)

# ----------------------------
# Gradio Interface
# ----------------------------

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 BiMER: Bimodal Emotion Recognition")
    gr.Markdown("Analyze emotions from **Speech** and **Text** using Wav2Vec 2.0 and RoBERTa.")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text Input", placeholder="Type something emotional...", lines=3)
            audio_input = gr.Audio(label="Audio Input (WAV)", type="filepath")
            submit_btn = gr.Button("Analyze Emotion", variant="primary")
        
        with gr.Column():
            output_text = gr.Markdown(label="Results")

    submit_btn.click(
        fn=predict_bimodal,
        inputs=[text_input, audio_input],
        outputs=output_text
    )
    
    gr.Markdown("---")
    gr.Markdown("### Developed by S R Gudlavalleru Engineering College Team")

if __name__ == "__main__":
    demo.launch(share=False)
