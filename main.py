import os
import pyaudio
import wave
import keyboard
import time
from groq import Groq
from dotenv import load_dotenv
import re
from pydub import AudioSegment
from playsound3 import playsound

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

def get_llm_response(text, api_key=None, stream=False):
    """
    Sends text to Groq's LLM and gets the response.
    
    Args:
        text: The text to send to the LLM
        api_key: Groq API key (will use environment variable if None)
        stream: Whether to stream the response or not
        
    Returns:
        LLM response text
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
    
    # Create the chat completion
    if stream:
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
    else:
        # Get complete response
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": text}
            ]
        )
        
        # Extract and return the response text
        response_text = response.choices[0].message.content
        
        # Print the response
        print("\n=== LLM Response ===")
        print(response_text)
        print("===================\n")
        
        return response_text

def split_text_into_chunks(text, max_chars=800):
    """
    Splits text into smaller chunks respecting paragraph and sentence boundaries.
    
    Args:
        text: The text to split
        max_chars: Maximum characters per chunk (approximation for token limit)
        
    Returns:
        List of text chunks
    """
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
                    # Handle very long sentences by breaking at reasonable points
                    sentence_parts = []
                    for i in range(0, len(sentence), max_chars - 100):
                        part = sentence[i:i + max_chars - 100].strip()
                        if part:
                            sentence_parts.append(part)
                    
                    for part in sentence_parts:
                        chunks.append(part)
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

def text_to_speech_chunked(text, api_key=None, output_filename="speech.wav"):
    """
    Converts text to speech using Groq's TTS API by splitting into chunks.
    
    Args:
        text: The text to convert to speech
        api_key: Groq API key (will use environment variable if None)
        output_filename: Path to save the final output audio file
        
    Returns:
        Path to the saved audio file
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
    
    # Split text into smaller chunks
    chunks = split_text_into_chunks(text)
    print(f"Text split into {len(chunks)} chunks for TTS processing")
    
    # Create directory for temporary files
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each chunk and save as temporary audio file
    temp_files = []
    
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
        
        return output_filename
    else:
        print("No audio chunks were created successfully.")
        return None

def play_audio(audio_file_path):
    """
    Plays an audio file using playsound3.
    
    Args:
        audio_file_path: Path to the audio file to play
    """
    try:
        print(f"Playing audio from {audio_file_path}...")
        playsound(audio_file_path)
        print("Audio playback completed.")
    except Exception as e:
        print(f"Error playing audio: {e}")

def main():
    """Main function to test audio recording, transcription, LLM response, and TTS."""
    # Get API key from environment
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("Warning: GROQ_API_KEY environment variable not set.")
        api_key = input("Please enter your Groq API key: ")
    
    # Save the recorded audio file in the current directory
    output_filename = os.path.join(os.getcwd(), "recorded_audio.wav")
    speech_filename = os.path.join(os.getcwd(), "llm_response.wav")
    
    try:
        # Record audio
        print("=== Real-time Audio Transcription, LLM Response, and Speech Synthesis with Groq API ===")
        print("This script will record audio, transcribe it, get an LLM response, and convert the response to speech.")
        
        # Ask if the user wants to stream the LLM response
        stream_choice = input("Do you want to stream the LLM response? (y/n): ").lower().strip()
        stream = stream_choice == 'y' or stream_choice == 'yes'
        
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
                
                # Send the transcription to the LLM
                llm_response = get_llm_response(transcription, api_key, stream)
                
                # Convert the LLM response to speech using chunking
                speech_file = text_to_speech_chunked(llm_response, api_key, speech_filename)
                
                # Play the synthesized speech if successful
                if speech_file:
                    play_audio(speech_file)
            else:
                print("Transcription cancelled by user.")
    
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
