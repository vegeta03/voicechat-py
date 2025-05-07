#!/usr/bin/env python3
"""
Real-time Audio Transcription, LLM Response, and Speech Synthesis with Groq API.

This script records audio, transcribes it using Groq's API, gets an LLM response,
and converts the response to speech.
"""

import os
import sys
import time
import wave
from pathlib import Path
from typing import List, Optional, Union

try:
    import numpy as np
    import pyaudio
    import keyboard
    from groq import Groq
    from dotenv import load_dotenv
    import re
    from pydub import AudioSegment
    from playsound3 import playsound
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Please install required packages with:")
    print("pip install numpy pyaudio keyboard groq python-dotenv pydub playsound3")
    sys.exit(1)

# Constants
DEFAULT_MAX_CHARS_PER_CHUNK = 800
DEFAULT_RECORD_KEY = "space"
DEFAULT_CANCEL_KEY = "esc"


def display_audio_level(level: float, width: int = 40) -> None:
    """Display audio level as a visual indicator."""
    # Scale to 0-1 range, applying some non-linear scaling for better visualization
    scaled_level = min(1.0, level * 5)  # Amplify for better visibility
    bar_length = int(scaled_level * width)
    
    # Create the bar with different colors for different levels
    if scaled_level < 0.3:
        bar = '\033[92m' + '█' * bar_length + '\033[0m'  # Green
    elif scaled_level < 0.7:
        bar = '\033[93m' + '█' * bar_length + '\033[0m'  # Yellow
    else:
        bar = '\033[91m' + '█' * bar_length + '\033[0m'  # Red
        
    # Fill the rest with empty space
    empty_space = ' ' * (width - bar_length)
    
    # Display the level bar, overwriting the previous line
    sys.stdout.write(f"\rLevel: [{bar}{empty_space}] {level:.2f}")
    sys.stdout.flush()


def load_environment() -> dict:
    """
    Load environment variables from .env file.
    
    Returns:
        Dictionary containing required API keys and configuration
    """
    load_dotenv(override=True)
    
    config = {
        "api_key": os.getenv("GROQ_API_KEY"),
        "stt_model": os.getenv("GROQ_STT", "whisper-large-v3-turbo"),
        "llm_model": os.getenv("GROQ_LLM", "llama-3.1-8b-instant"),
        "tts_model": os.getenv("GROQ_TTS", "playai-tts"),
        "max_chars": int(os.getenv("MAX_CHARS_PER_CHUNK", DEFAULT_MAX_CHARS_PER_CHUNK))
    }
    
    print(f"GROQ_API_KEY loaded: {'✓' if config['api_key'] else '✗'}")
    
    return config


def record_audio(output_filename: str, key_to_hold: str = DEFAULT_RECORD_KEY, 
                 cancel_key: str = DEFAULT_CANCEL_KEY) -> bool:
    """
    Records audio while the specified key is held down, with audio level visualization.
    
    Args:
        output_filename: Path to save the recorded audio
        key_to_hold: Key that must be held to record (default: space)
        cancel_key: Key to press to cancel recording (default: esc)
        
    Returns:
        Boolean indicating if recording was successful
    """
    # Audio parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # 16kHz is optimal for speech recognition
    CHUNK = 1024
    
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Prepare the stream
    try:
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        audio.terminate()
        return False
    
    print(f"Hold the {key_to_hold.upper()} key to start recording...")
    print(f"Press '{cancel_key}' to cancel recording")
    
    # Wait for key press
    try:
        while not keyboard.is_pressed(key_to_hold):
            if keyboard.is_pressed(cancel_key):
                print("Recording cancelled before it started.")
                stream.stop_stream()
                stream.close()
                audio.terminate()
                return False
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        return False
    
    print("Recording started! Keep holding the key...")
    
    # Record while key is held
    frames = []
    try:
        while keyboard.is_pressed(key_to_hold):
            if keyboard.is_pressed(cancel_key):
                print("\nRecording cancelled.")
                break
                
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Calculate audio level for visualization
            audio_data = np.frombuffer(data, dtype=np.int16)
            level = np.abs(audio_data).mean() / 32767.0  # Normalize to 0-1
            display_audio_level(level)
            
    except Exception as e:
        print(f"\nError during recording: {e}")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("\nRecording stopped.")
    
    # Save the recorded audio to WAV file
    if frames:
        try:
            with wave.open(output_filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            print(f"Audio saved to {output_filename}")
            return True
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False
    else:
        print("No audio recorded.")
        return False


def transcribe_with_groq(audio_file_path: str, api_key: Optional[str] = None) -> str:
    """
    Transcribes audio using Groq's Whisper API.
    
    Args:
        audio_file_path: Path to the audio file
        api_key: Groq API key (will use environment variable if None)
        
    Returns:
        Transcription text
        
    Raises:
        ValueError: If API key is not provided
        Exception: If transcription fails
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
    try:
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model=model,  # Use the model from environment variable
                response_format="json",
                language="en"
            )
        
        return transcription.text
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise


def get_llm_response(text: str, api_key: Optional[str] = None) -> str:
    """
    Sends text to Groq's LLM and gets the response.
    
    Args:
        text: The text to send to the LLM
        api_key: Groq API key (will use environment variable if None)
        
    Returns:
        LLM response text
        
    Raises:
        ValueError: If API key is not provided
        Exception: If LLM request fails
    """
    # Use provided API key or get from environment
    api_key = api_key or os.getenv("GROQ_API_KEY")
    
    # Get model from environment variable or use default
    model = os.getenv("GROQ_LLM", "llama-3.1-8b-instant")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not provided or set in environment")
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    print(f"Sending text to Groq LLM ({model})...")
    
    try:
        # Stream the response
        print("\n=== LLM Response (Streaming) ===")
        full_response = ""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": text}
            ],
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                
        print("\n===============================\n")
        return full_response
    except Exception as e:
        print(f"Error getting LLM response: {e}")
        raise


def split_text_into_chunks(text: str, max_chars: int = DEFAULT_MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Splits text into smaller chunks respecting word boundaries.
    
    Args:
        text: The text to split
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Get the max_chars from environment if available
    env_max_chars = os.getenv("MAX_CHARS_PER_CHUNK")
    if env_max_chars:
        try:
            max_chars = int(env_max_chars)
            print(f"Using MAX_CHARS_PER_CHUNK={max_chars} from environment")
        except ValueError:
            print(f"Invalid MAX_CHARS_PER_CHUNK value: {env_max_chars}, using default {max_chars}")
    
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph is too long, split by sentences
        if len(paragraph) > max_chars:
            # Add any existing chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            # Process each sentence
            for sentence in sentences:
                if len(sentence) > max_chars:
                    # Handle very long sentences by breaking at word boundaries
                    words = sentence.split()
                    current_sentence = ""
                    
                    for word in words:
                        if len(current_sentence) + len(word) + 1 > max_chars:
                            if current_sentence:  # Avoid empty chunks
                                chunks.append(current_sentence.strip())
                            current_sentence = word
                        else:
                            if current_sentence:
                                current_sentence += " " + word
                            else:
                                current_sentence = word
                    
                    if current_sentence:  # Add any remaining text
                        chunks.append(current_sentence.strip())
                else:
                    if len(current_chunk) + len(sentence) + 1 > max_chars:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        if current_chunk:
                            current_chunk += " " + sentence
                        else:
                            current_chunk = sentence
        else:
            # If adding this paragraph would exceed the limit, start a new chunk
            if len(current_chunk) + len(paragraph) + 2 > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
    
    # Add the final chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def text_to_speech_chunked(text: str, api_key: Optional[str] = None, 
                           output_filename: str = "speech.wav") -> Optional[str]:
    """
    Converts text to speech using Groq's TTS API by splitting into chunks.
    
    Args:
        text: The text to convert to speech
        api_key: Groq API key (will use environment variable if None)
        output_filename: Path to save the final output audio file
        
    Returns:
        Path to the saved audio file or None if unsuccessful
        
    Raises:
        ValueError: If API key is not provided
    """
    # Use provided API key or get from environment
    api_key = api_key or os.getenv("GROQ_API_KEY")
    
    # Get model from environment variable
    model = os.getenv("GROQ_TTS", "playai-tts")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not provided or set in environment")
    
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    print("\n=== Converting LLM response to speech ===")
    print(f"Sending text to Groq TTS ({model})...")
    
    # Split text into smaller chunks using environment variable for max_chars
    chunks = split_text_into_chunks(text)
    print(f"Text split into {len(chunks)} chunks for TTS processing")
    
    # Create directory for temporary files
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each chunk and save as temporary audio file
    temp_files = []
    
    try:
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} characters)")
            temp_file = os.path.join(temp_dir, f"chunk_{i}.wav")
            
            try:
                # Default voice
                voice = "Arista-PlayAI"
                
                # Create the speech
                response = client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=chunk,
                    response_format="wav"
                )
                
                # Save the response to a file
                response.write_to_file(temp_file)
                temp_files.append(temp_file)
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {e}")
        
        # Combine all audio files into one using pydub
        if temp_files:
            print("Combining audio chunks...")
            
            # Create a combined audio file
            combined = AudioSegment.empty()
            
            for temp_file in temp_files:
                segment = AudioSegment.from_file(temp_file)
                combined += segment
            
            # Export the combined audio
            combined.export(output_filename, format="wav")
            print(f"Combined TTS response saved to {output_filename}")
            
            return output_filename
        else:
            print("No audio chunks were created successfully.")
            return None
            
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")
        return None
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        
        try:
            os.rmdir(temp_dir)
        except:
            pass


def play_audio(audio_file_path: str, cancel_key: str = DEFAULT_CANCEL_KEY) -> None:
    """
    Plays an audio file using playsound3 with ability to cancel playback.
    
    Args:
        audio_file_path: Path to the audio file to play
        cancel_key: Key to press to cancel playback (default: 'esc')
    """
    try:
        print(f"Playing audio from {audio_file_path}...")
        print(f"Press '{cancel_key}' to stop playback")
        
        # Play sound non-blocking
        sound = playsound(audio_file_path, block=False)
        
        # Monitor for cancel key
        while sound.is_alive():
            if keyboard.is_pressed(cancel_key):
                sound.stop()
                print("\nPlayback cancelled.")
                break
            time.sleep(0.1)
            
        if not sound.is_alive():
            print("Audio playback completed.")
            
    except Exception as e:
        print(f"Error playing audio: {e}")


def main() -> int:
    """
    Main function to handle audio recording, transcription, LLM response, and TTS.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load environment variables and configuration
    config = load_environment()
    
    # Check for API key
    api_key = config["api_key"]
    if not api_key:
        print("Warning: GROQ_API_KEY environment variable not set.")
        api_key = input("Please enter your Groq API key: ")
        if not api_key:
            print("Error: API key is required to use Groq services.")
            return 1
    
    # Define file paths
    output_filename = os.path.join(os.getcwd(), "recorded_audio.wav")
    speech_filename = os.path.join(os.getcwd(), "llm_response.wav")
    
    try:
        # Print introduction
        print("=== Real-time Audio Transcription, LLM Response, and Speech Synthesis with Groq API ===")
        print("This script will record audio, transcribe it, get an LLM response, and convert the response to speech.")
        print(f"Press '{DEFAULT_RECORD_KEY}' to record and '{DEFAULT_CANCEL_KEY}' to cancel at any stage.")
        
        # Record audio when space bar is held down
        success = record_audio(output_filename)
        
        if success:
            # Start the timer before STT call
            start_time = time.perf_counter()
            
            # Transcribe the recorded audio
            transcription = transcribe_with_groq(output_filename, api_key)
            
            # Display the transcription
            print("\n=== Transcription Result ===")
            print(transcription)
            print("============================\n")
            
            # Send the transcription to the LLM
            llm_response = get_llm_response(transcription, api_key)
            
            # Convert the LLM response to speech using chunking
            speech_file = text_to_speech_chunked(llm_response, api_key, speech_filename)
            
            # Stop the timer after TTS processing
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            # Print the total time
            print(f"\n=== Total Processing Time ===")
            print(f"Time from STT to TTS completion: {elapsed_time:.4f} seconds")
            print("==============================\n")
            
            # Play the synthesized speech if successful
            if speech_file:
                play_audio(speech_file)
                
            return 0
        else:
            print("Recording was not successful.")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Error occurred: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
