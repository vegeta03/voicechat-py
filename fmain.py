import os
import pyaudio
import wave
import keyboard
import asyncio
import logging
import time
import concurrent.futures
from groq import Groq, AuthenticationError
from dotenv import load_dotenv
import traceback

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Load environment variables from .env file
load_dotenv(override=True)
print(f"GROQ_API_KEY loaded: {'✓' if os.getenv('GROQ_API_KEY') else '✗'}")


# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.is_recording = False
        
    def start_recording(self):
        self.is_recording = True
        self.frames = []
        
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("Recording started! Keep holding the key...")
        while self.is_recording and keyboard.is_pressed("space"):
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
                time.sleep(0.01) 
            except Exception as e:
                logging.error(f"Error reading audio stream: {e}")
                self.is_recording = False
                break
        self.stop_recording()
        
    def stop_recording(self):
        self.is_recording = False
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logging.error(f"Error stopping/closing audio stream: {e}")
            finally:
                self.stream = None
        print("Recording stopped.")
    
    def save_to_file(self, filename):
        if not self.frames:
            print("No frames recorded to save.")
            return False
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.frames))
            print(f"Audio saved to {filename}")
            return True
        except Exception as e:
            logging.error(f"Error saving audio to file {filename}: {e}")
            return False
        
    def __del__(self):
        if hasattr(self, 'stream') and self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception: 
                pass 
        if hasattr(self, 'audio') and self.audio:
            try:
                self.audio.terminate()
            except Exception:
                pass

class GroqClient:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.stt_model = os.environ.get("GROQ_STT")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please ensure it is set in your .env file (e.g., GROQ_API_KEY=your_key) "
                "and that the .env file is located in the script's working directory. "
                "The `python-dotenv` library (call `load_dotenv()`) should be used to load it."
            )
        self.client = Groq(api_key=self.api_key)
    
    def transcribe(self, audio_file):
        print("Sending audio to Groq API for transcription...")
        with open(audio_file, "rb") as file_obj: # Renamed to file_obj to avoid conflict
            transcription_response = self.client.audio.transcriptions.create( # Renamed variable
                file=(os.path.basename(audio_file), file_obj), 
                model=self.stt_model,
                response_format="json",
                language="en"
            )
        if hasattr(transcription_response, 'text'):
            return transcription_response.text
        else:
            logging.warning("Transcription response does not have a 'text' attribute. Full response: %s", transcription_response)
            return str(transcription_response)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, lambda: func(*args, **kwargs)
    )

async def wait_for_key_press(key_name: str):
    while True:
        if keyboard.is_pressed(key_name):
            await asyncio.sleep(0.05) 
            if keyboard.is_pressed(key_name):
                 return True
        await asyncio.sleep(0.01)

async def get_confirmation():
    print(f"Do you want to send the recorded audio file to Groq for transcription?")
    # MODIFIED: Prompt wording changed
    print("Press Enter to confirm or any other key to cancel:") 
    try:
        response = await run_in_thread(input)
        return response.strip() == ""
    except Exception as e:
        logging.error(f"Error getting confirmation: {e}")
        return False

async def main():
    loop = asyncio.get_running_loop()
    loop.slow_callback_duration = 1.0 
    
    output_path = os.path.join(os.getcwd(), "recorded_audio.wav")
    recorder = AudioRecorder()
    
    try:
        print("Hold the SPACE key to start recording...")
        await wait_for_key_press("space") 
        
        await run_in_thread(recorder.start_recording) 
        
        success = recorder.save_to_file(output_path)
        if not success:
            print("Recording or saving failed.")
            return
        
        confirmed = await get_confirmation()
        if not confirmed:
            print("Transcription cancelled by user.")
            return
        
        client = GroqClient() 
        transcription = await run_in_thread(client.transcribe, output_path)
        
        if transcription is not None:
            print("\n=== Transcription Result ===")
            print(transcription)
            print("============================\n")
        else:
            print("Transcription failed or returned no text.")
            
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
        print("Please ensure your .env file is correctly set up with a valid GROQ_API_KEY.")
    # MODIFIED: Correctly catching AuthenticationError
    except AuthenticationError as authe: 
        print(f"Groq Authentication Error: {authe}")
        print("This means your GROQ_API_KEY is invalid or expired. Please check your API key in the .env file and on the Groq Cloud dashboard.")
    except Exception as e:
        print(f"An unexpected program error occurred: {e}")
        traceback.print_exc()
    finally:
        print("Shutting down thread pool executor...")
        executor.shutdown(wait=True) 
        print("Program finished.")

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print(
            "Warning: GROQ_API_KEY not found in environment before asyncio.run(). "
            "Ensure load_dotenv() has been called at the module level and .env is correct."
        )
    asyncio.run(main())
