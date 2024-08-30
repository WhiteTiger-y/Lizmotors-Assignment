### README.md for `lizmotors_assignment_sts_llm_with_interface.ipynb`

# AI Assistant with Whisper, Bark, and Gradio

This project implements an AI-powered voice assistant that utilizes Whisper for speech-to-text, an LLM for generating responses, and Bark for converting text responses back into speech. The interface is built using Gradio for easy interaction, making it accessible via a web interface.

---

## Project Overview

The voice assistant follows a structured workflow:
1. **Audio Input**: Users record or upload an audio file.
2. **Speech-to-Text Conversion**: Whisper processes the audio and converts it to text.
3. **Text Generation**: The transcribed text is passed to a Large Language Model (LLM), which generates a conversational response.
4. **Text-to-Speech Conversion**: Bark converts the generated text response into audio, allowing users to hear the AI's response.
5. **Interactive Interface**: The entire process is handled through an intuitive Gradio interface where users can select speakers, upload audio, and listen to responses.

---

## Workflow Architecture

### 1. **Speech-to-Text (Whisper)**
   - **Whisper Model**: We utilize the `whisper` library from OpenAI, specifically the "medium" model, for speech recognition.
   - **Functionality**: The `transcribe()` function handles the conversion of audio to text by processing audio files through Whisper's pre-trained model.
   - **Implementation**:
     - Audio files are loaded and preprocessed using `librosa`.
     - The Whisper model transcribes the audio by converting it into a mel spectrogram and decoding it into text.

### 2. **Text Generation (LLM)**
   - **Model Used**: We leverage `google/flan-t5-xl`, a pre-trained language model for text generation.
   - **Quantization**: To optimize performance, the model is loaded with 4-bit quantization via `BitsAndBytesConfig`.
   - **Functionality**: The `pipe` object handles the text generation process, where the transcribed text is fed into the model and a conversational response is generated.
   - **Instruction to LLM**: The assistant is instructed to answer user queries in an accurate and concise manner, with detailed explanations where necessary.

### 3. **Text-to-Speech (Bark)**
   - **Bark Model**: Suno’s Bark is used for high-quality text-to-speech synthesis.
   - **Speaker Selection**: Users can choose from different pre-configured voices (e.g., `english-male-1`, `english-male-2`, `english-female`).
   - **Functionality**: The `text_to_speech_bark()` function converts the LLM-generated response into an audio file using the Bark model’s `generate_audio()` function.
   - **Audio Output**: The generated audio is saved as a `.wav` file and played back to the user.

### 4. **Gradio Interface**
   - **User Interface**: Gradio provides an interactive web-based interface that allows users to:
     1. Upload or record audio.
     2. Select a speaker for TTS.
     3. Process the audio to receive a text response.
     4. Listen to the AI-generated audio response.
   - **Buttons**:
     - `Process Audio`: Executes the full pipeline from STS to TTS.
     - `Clear`: Resets the inputs and outputs to allow for a new interaction.

---

## Flow Diagram

Below is a conceptual flow diagram that describes the overall architecture and workflow of the voice assistant.

```mermaid
graph TD
    A[User] -->|Upload Audio| B[Gradio Interface]
    B -->|Audio Input| C[Whisper Model]
    C -->|Text Output| D[LLM (Flan-T5)]
    D -->|Generated Text| E[Bark TTS]
    E -->|Audio Output| B
```

This simplified `mermaid` diagram should now render correctly.

1. **User Interaction**: The user uploads or records audio in the Gradio interface.
2. **Speech-to-Text Conversion**: Whisper processes the audio to convert it into text.
3. **Text Generation**: The LLM processes the transcribed text and generates a response.
4. **Text-to-Speech Conversion**: Bark converts the LLM-generated text to audio using a selected speaker.
5. **Output**: The processed audio response is played back to the user.

---

## Code Breakdown

### 1. **Setup and Installation**
   - The necessary Python packages are installed using `pip`. This includes `transformers`, `whisper`, `gradio`, `bark`, and others.

### 2. **Model Loading**
   - **Whisper Model**: Loaded using `whisper.load_model("medium")`.
   - **Text Generation Model**: Loaded using Hugging Face’s `pipeline` with 4-bit quantization for efficiency.
   - **Bark Model**: Loaded via `AutoProcessor` and `AutoModel` for text-to-speech synthesis.

### 3. **Core Functions**
   - **convert_audio_to_whisper_format()**: Converts audio to the format required by Whisper.
   - **transcribe()**: Handles speech-to-text conversion.
   - **text_to_speech_bark()**: Converts text to speech using the selected Bark speaker.
   - **process_audio()**: Orchestrates the entire pipeline (STS → LLM → TTS).

### 4. **Gradio Interface**
   - **Blocks API**: Gradio’s `Blocks` API is used to define the layout and interaction logic of the interface.
   - **Components**: The interface includes an audio uploader, a dropdown for speaker selection, textboxes for displaying results, and buttons for processing or clearing inputs.

---

## Running the Notebook

1. **Environment Setup**: The notebook is designed to run on Google Colab with an A100 GPU for optimal performance.
2. **Execution**: Run all cells sequentially to set up the environment, load models, and launch the Gradio interface.
3. **Accessing the Interface**: In Colab, you will receive a `gradio.live` link to access the web-based interface for interacting with the assistant.

---

## Dependencies

Ensure you have the following dependencies installed in your environment. These can be installed using `pip`:

```bash
transformers==4.37.2
bitsandbytes==0.41.3
accelerate==0.25.0
whisper
gradio
gTTS
huggingface_hub
bark
librosa
soundfile
```
This README provides detailed information about the architecture, workflow, and usage of the AI-powered voice assistant implemented in the `lizmotors_assignment_sts_llm_with_interface.ipynb` notebook.
