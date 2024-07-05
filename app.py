from flask import Flask, request, render_template, jsonify
import os
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import tempfile

# Flask app setup
app = Flask(__name__)

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Define model ID
model_id = "openai/whisper-large-v3"

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

# Set up pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Function to transcribe audio file
def transcribe_audio(file_path):
    audio, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    result = pipe(audio)
    return result['text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['audio_data']
    if file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            file.save(tmp.name)
            transcription = transcribe_audio(tmp.name)
            os.remove(tmp.name)  # Clean up the temporary file
            return jsonify({'transcription': transcription})
    return jsonify({'error': 'No audio data received'}), 400

if __name__ == '__main__':
    app.run(debug=True)
