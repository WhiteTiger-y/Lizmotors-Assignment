# Documentation for Voice-to-Voice with Customizable Voice

## Overview

This project creates a Gradio interface that allows users to record their voice, convert it to text using Whisper, and then synthesize speech from the text using the Bark model. Users can select different voice presets for the speech synthesis.

## Dependencies

1. **Transformers**: Provides state-of-the-art models for text generation.
2. **BitsAndBytes**: Enables quantization of models to reduce their size.
3. **Whisper**: A model for high-quality speech-to-text conversion.
4. **Gradio**: A library to create easy-to-use interfaces for machine learning models.
5. **Librosa**: A library for audio and music analysis.
6. **Soundfile**: A library for reading and writing audio files.
7. **Bark**: Provides models for text-to-speech generation.

### Installation

To install the required packages, use the following commands:

```bash
!pip install -q transformers==4.37.2
!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
!pip install -q git+https://github.com/openai/whisper.git
!pip install -q gradio
!pip install -q huggingface_hub
!pip install bark
```

## Code Explanation

### Imports

```python
import torch
from transformers import AutoProcessor, AutoModel
import whisper
import gradio as gr
import numpy as np
import librosa
import soundfile as sf
from bark import generate_audio, SAMPLE_RATE
import warnings
import os
import wave
from io import BytesIO  # For handling audio file in memory
```

- **torch**: A library for tensor computation and deep learning.
- **transformers**: Provides models and processors for text generation.
- **whisper**: Used for converting speech to text.
- **gradio**: For creating a user interface.
- **numpy**: For numerical operations.
- **librosa**: For loading and processing audio files.
- **soundfile**: For reading and writing audio files.
- **bark**: For generating speech from text.
- **warnings**: To handle and suppress warnings.
- **os**: For operating system-dependent functionality.
- **wave**: For handling WAV audio files.
- **BytesIO**: Allows handling files as byte streams in memory.

### Device Configuration

```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using torch {torch.__version__} ({DEVICE})")
```

- **DEVICE**: Determines whether to use a GPU (`cuda`) or CPU based on availability.

### Model Loading

**Whisper Model for Speech-to-Text:**

```python
model_whisper = whisper.load_model("medium", device=DEVICE)
```

- **whisper.load_model("medium")**: Loads the Whisper model, which is used for converting spoken language into text. The `"medium"` variant is a balanced model between performance and computational cost.

**Text Generation Model:**

```python
model_id = "google/flan-t5-large"
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipe = pipeline(
    "text2text-generation",
    model=model_id,
    model_kwargs={"quantization_config": quant_config}
)
print(f"Loaded Text Generation model: {model_id}")
```

- **model_id**: Identifier for the text generation model.
- **BitsAndBytesConfig**: Configuration for loading the model in 4-bit precision to save memory.
- **pipeline**: Creates a text-to-text generation pipeline using the specified model.

**Bark Model and Processor:**

```python
processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")
```

- **AutoProcessor** and **AutoModel**: Load the Bark model and its associated processor for text-to-speech conversion.

### Speaker Options

```python
SPEAKERS = {
    "english-male-1": "v2/en_speaker_1",
    "english-male-2": "v2/en_speaker_2",
    "english-female": "v2/en_speaker_9",
    "hindi-male-1": "v2/hi_speaker_2",
    "hindi-male-2": "v2/hi_speaker_5",
    "hindi-female-1": "v2/hi_speaker_0",
    "hindi-female-2": "v2/hi_speaker_4"
}
```

- **SPEAKERS**: A dictionary mapping speaker names to their respective voice presets used in Bark.

### Functions

**Convert Audio to Whisper Format:**

```python
def convert_audio_to_whisper_format(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    processed_audio_path = "processed_audio.wav"
    sf.write(processed_audio_path, audio, sr)
    return processed_audio_path
```

- **librosa.load**: Loads the audio file and resamples it to 16 kHz.
- **sf.write**: Saves the processed audio to a temporary WAV file for transcription.

**Transcribe Audio:**

```python
def transcribe(audio_path):
    if audio_path is None:
        return ''  # Return empty string if no audio input

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
    result = model_whisper.decode(mel)
    return result.text
```

- **whisper.load_audio**: Loads the audio for Whisper processing.
- **whisper.pad_or_trim**: Pads or trims the audio to match the model's expected input length.
- **whisper.log_mel_spectrogram**: Converts audio to a mel spectrogram.
- **model_whisper.decode**: Decodes the mel spectrogram to text.

**Generate Text-to-Speech:**

```python
def text_to_speech(text, speaker):
    voice_presets = {
        "english-male-1": "v2/en_speaker_1",
        "english-male-2": "v2/en_speaker_2",
        "english-female": "v2/en_speaker_9",
        "hindi-male-1": "v2/hi_speaker_2",
        "hindi-male-2": "v2/hi_speaker_5",
        "hindi-female-1": "v2/hi_speaker_0",
        "hindi-female-2": "v2/hi_speaker_4"
    }

    voice_preset = voice_presets.get(speaker, "v2/en_speaker_1")
    inputs = processor(text, voice_preset=voice_preset)
    audio_array = model.generate(**inputs)
    audio_array = audio_array.cpu().numpy().squeeze()  # Convert to numpy array and remove unnecessary dimensions
    return audio_array, SAMPLE_RATE
```

- **voice_presets**: Maps speaker options to their corresponding voice presets.
- **processor**: Processes the text input with the selected voice preset.
- **model.generate**: Generates audio from the processed text.
- **audio_array.cpu().numpy().squeeze()**: Converts the generated audio to a numpy array and removes any extra dimensions.

**Process Audio Function:**

```python
def process_audio(audio_file, speaker):
    if isinstance(audio_file, BytesIO):
        audio_file = BytesIO(audio_file.read())
    
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getvalue())

    processed_audio_path = convert_audio_to_whisper_format(temp_audio_path)
    transcription = transcribe(processed_audio_path)
    audio_array, sample_rate = text_to_speech(transcription, speaker)
    
    output_audio_path = "output_speech.wav"
    sf.write(output_audio_path, audio_array, sample_rate)
    
    return transcription, output_audio_path
```

- **BytesIO**: Handles audio data in memory if received as a byte stream.
- **convert_audio_to_whisper_format**: Converts and saves the audio file for transcription.
- **transcribe**: Converts the audio to text.
- **text_to_speech**: Generates speech from the transcribed text.
- **sf.write**: Saves the generated speech to a file.

### Gradio Interface

```python
interface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(source="microphone", type="file", label="Record Audio"),  # Microphone input
        gr.Dropdown(
            label="Select Speaker",
            choices=list(SPEAKERS.keys()),
            value="english-male-1"  # Default speaker
        )
    ],
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Audio(label="Generated Speech")
    ],
    title="Voice-to-Voice with Customizable Voice",
    description="Record your voice and select a speaker to generate speech based on the transcribed text."
)

# Launch the Gradio interface
interface.launch(debug=True)
```

- **gr.Interface**: Creates a web interface for interacting with the functions.
- **fn**: Specifies the function (`process_audio`) to be called when inputs are provided.
- **inputs**: Defines the input components:
  - **gr.Audio

**: Allows recording audio from the microphone.
  - **gr.Dropdown**: Lets users select a speaker.
- **outputs**: Defines the output components:
  - **gr.Textbox**: Displays the transcribed text.
  - **gr.Audio**: Plays the generated speech.
- **title** and **description**: Provide information about the interface.

## Usage

1. **Record Audio**: Use the microphone input to record your voice.
2. **Select Speaker**: Choose the desired voice preset from the dropdown menu.
3. **Generate Output**: The system will transcribe the audio and synthesize speech based on the transcription. The results will be displayed as text and audio output.

## Notes

- Ensure microphone permissions are enabled in the browser for recording.
- A stable internet connection is required for model loading and inference.
