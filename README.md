# VoiceChat-py

A real-time voice chat application that uses Groq's API for speech-to-text, language model processing, and text-to-speech capabilities.

## Overview

VoiceChat-py is a Python-based application that allows you to:

1. Record audio by holding down a key (default: spacebar)
2. Transcribe the audio using Groq's Whisper API
3. Process the transcription with a language model (LLM)
4. Convert the LLM's response back to speech
5. Play the audio response

This creates a seamless voice conversation experience with AI.

## Features

- **Real-time audio recording**: Record audio while holding down a key
- **Speech-to-text**: Transcribe audio using Groq's Whisper API
- **Language model processing**: Process transcriptions with Groq's LLM API
- **Text-to-speech**: Convert LLM responses to speech using Groq's TTS API
- **Configurable**: Customize models and parameters via environment variables

## Requirements

- Python 3.7+
- Groq API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/voicechat-py.git
   cd voicechat-py
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Groq API key and configuration:

   ```bash
   cp .env.sample .env
   ```

   Then edit the `.env` file to add your actual Groq API key.

## Usage

### Basic Usage

Run the main script:

```bash
python main.py
```

- Hold down the spacebar to record audio
- Release to stop recording and start processing
- The application will transcribe your audio, send it to the LLM, and play back the response

## Configuration

You can customize the application behavior through environment variables in the `.env` file:

- `GROQ_API_KEY`: Your Groq API key (required)
- `GROQ_STT`: Speech-to-text model (default: "distil-whisper-large-v3-en")
- `GROQ_LLM`: Language model (default: "llama-3.1-8b-instant")
- `GROQ_TTS`: Text-to-speech model (default: "playai-tts")
- `MAX_CHARS_PER_CHUNK`: Maximum characters per TTS chunk (default: 1200)

## Project Structure

- `main.py`: Main script with all functionality combined
- `requirements.txt`: Required Python packages
- `.env.sample`: Sample environment configuration

## Acknowledgements

- This project uses the [Groq API](https://groq.com) for AI capabilities
- Built with [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio recording
