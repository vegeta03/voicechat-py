import os
import pyaudio
import wave
import keyboard
import time
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
print(f"GROQ_API_KEY loaded: {'✓' if os.getenv('GROQ_API_KEY') else '✗'}")

def record_audio(output_filename, key_to_hold="space"):
    """
    Records audio while the specified key is held down.
    Args:
        output_filename: Path to save the recorded audio
        key_to_hold: Key that must be held to record (default: space)
    """
    # Audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # 16kHz is optimal for speech recognition
    CHUNK = 1024

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Prepare the stream
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print(f"Hold the {key_to_hold.upper()} key to start recording...")
    # Wait for key press
    while not keyboard.is_pressed(key_to_hold):
        time.sleep(0.1)
    
    print("Recording started! Keep holding the key...")

    # Record while key is held
    frames = []
    try:
        while keyboard.is_pressed(key_to_hold):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("Recording stopped.")

    # Save the recorded audio to WAV file
    if frames:
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        print(f"Audio saved to {output_filename}")
        return True
    else:
        print("No audio recorded.")
        return False

def transcribe_with_groq(audio_file_path, api_key=None):
    """
    Transcribes audio using Groq's Whisper API.
    Args:
        audio_file_path: Path to the audio file
        api_key: Groq API key (will use environment variable if None)
    Returns:
        Transcription text
    """
    # Use provided API key or get from environment
    api_key = api_key or os.getenv("GROQ_API_KEY")

    # Get model from environment variable
    model = os.getenv("GROQ_STT", "whisper-large-v3-turbo")

    if not api_key:
        raise ValueError("GROQ_API_KEY not provided or set in environment")

    # Initialize Groq client
    client = Groq(api_key=api_key)

    print("Sending audio to Groq API for transcription...")

    # Open the audio file and transcribe
    with open(audio_file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=file,
            model=model,  # Use the model from environment variable
            response_format="json",
            language="en"
        )

    return transcription.text

def main():
    """Main function to test audio recording and transcription."""
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("Warning: GROQ_API_KEY environment variable not set.")
        api_key = input("Please enter your Groq API key: ")

    # Save the recorded audio file in the current directory
    output_filename = os.path.join(os.getcwd(), "recorded_audio.wav")

    try:
        # Record audio
        print("=== Real-time Audio Transcription with Groq API ===")
        print("This script will record audio and transcribe it using Groq's Whisper API.")
        
        # Record audio when space bar is held down
        success = record_audio(output_filename)
        
        if success:
            # Confirmation prompt
            print("Do you want to send the recorded audio file to Groq for transcription?")
            print("Press Enter to confirm or any other key to cancel:")
            key = input()
            if not key.strip():  # More robust check for Enter key (empty input)
                # Transcribe the recorded audio
                transcription = transcribe_with_groq(output_filename, api_key)
                # Display the transcription
                print("\n=== Transcription Result ===")
                print(transcription)
                print("============================\n")
            else:
                print("Transcription cancelled by user.")
                
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
