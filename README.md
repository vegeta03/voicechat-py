# VoiceChat-py

A real-time voice chat application that uses Groq's API for speech-to-text, language model processing, and text-to-speech capabilities. Supports both Groq and SmallestAI for TTS.

## Overview

VoiceChat-py is a Python-based CLI tool that enables seamless voice conversations with AI. It allows you to:

1. Record audio by holding a key (default: spacebar)
2. Transcribe the audio using Groq's Whisper API
3. Process the transcription with a language model (LLM)
4. Convert the LLM's response back to speech (Groq or SmallestAI)
5. Play the audio response (with cancel option)

## Features

- **Real-time audio recording**: Record audio while holding down a key (default: spacebar)
- **Audio level visualization**: See live audio input levels in your terminal
- **Speech-to-text**: Transcribe audio using Groq's Whisper API
- **Language model processing**: Process transcriptions with Groq's LLM API
- **Text-to-speech**: Convert LLM responses to speech using Groq's TTS API or SmallestAI
- **Configurable**: Customize models, providers, and parameters via environment variables
- **Flexible TTS provider**: Choose between Groq and SmallestAI for text-to-speech
- **Audio playback with cancel**: Cancel playback at any time with a key (default: Esc)
- **Clear error handling**: Helpful messages if dependencies or API keys are missing

## Requirements

- Python 3.7+
- Groq API key (required for all modes)
- (Optional) SmallestAI API key (for SmallestAI TTS)
- Audio input/output devices (microphone and speakers)
- The following Python packages:
  - numpy
  - pyaudio
  - keyboard
  - groq
  - python-dotenv
  - pydub
  - playsound3

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/vegeta03/voicechat-py.git
   cd voicechat-py
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the sample environment file and edit it:

   ```bash
   cp .env.sample .env
   ```

   Then edit `.env` to add your actual API keys and configuration.

## Usage

Run the main script:

```bash
python main.py
```

- **To record audio**: Hold down the spacebar (or configured key)
- **To stop recording**: Release the key
- **To cancel recording**: Press Esc (or configured cancel key)
- The application will transcribe your audio, send it to the LLM, convert the response to speech, and play back the response.
- **To cancel playback**: Press Esc during playback

### Switching TTS Provider

By default, Groq is used for text-to-speech. To use SmallestAI, set `TTS_PROVIDER=smallestai` in your `.env` and provide your SmallestAI API key.

## Configuration

All configuration is via environment variables (set in `.env`).

| Variable            | Description                                       | Default                    |
| ------------------- | ------------------------------------------------- | -------------------------- |
| GROQ_API_KEY        | Your Groq API key (required)                      |                            |
| GROQ_STT            | Speech-to-text model name                         | distil-whisper-large-v3-en |
| GROQ_LLM            | LLM model name                                    | llama-3.1-8b-instant       |
| GROQ_TTS            | TTS model name (Groq only)                        | playai-tts                 |
| MAX_CHARS_PER_CHUNK | Max chars per TTS chunk                           | 800                        |
| TTS_PROVIDER        | TTS provider: 'groq' or 'smallestai'              | groq                       |
| SMALLESTAI_API_KEY  | Your SmallestAI API key (if using SmallestAI TTS) |                            |
| SMALLESTAI_MODEL    | SmallestAI TTS model name                         | lightning                  |
| VOICE_ID            | Voice ID (for SmallestAI, e.g., emily)            | emily                      |
| RECORD_KEY          | Key to hold for recording audio                   | space                      |
| CANCEL_KEY          | Key to cancel recording/playback                  | esc                        |

You can add or update these in your `.env` file.

## Project Structure

- `main.py` — Main script with all functionality
- `requirements.txt` — Python dependencies
- `.env.sample` — Sample environment configuration

## Troubleshooting & FAQ

**Q: I get an ImportError for a missing package.**

- A: Install all dependencies with `pip install -r requirements.txt`.

**Q: The script says my API key is missing.**

- A: Make sure you have set your `GROQ_API_KEY` (and `SMALLESTAI_API_KEY` if needed) in your `.env` file.

**Q: Audio recording/playback doesn't work.**

- A: Check your microphone and speaker devices. Make sure they are enabled and accessible.

**Q: How do I switch to SmallestAI for TTS?**

- A: Set `TTS_PROVIDER=smallestai` and provide your `SMALLESTAI_API_KEY` in `.env`.

**Q: Can I use a different LLM or TTS model?**

- A: Yes, set `GROQ_LLM`, `GROQ_TTS`, or `SMALLESTAI_MODEL` as desired.

## Acknowledgements

- This project uses the [Groq API](https://groq.com) for AI capabilities
- Built with [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio recording
- SmallestAI TTS support via [SmallestAI](https://smallest.ai/) (if enabled)
