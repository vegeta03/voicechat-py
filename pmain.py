#!/usr/bin/env python3
"""
Real-time Audio Transcription Tool

This script records audio while a key is held down and transcribes it using Groq's Whisper API.
"""

import os
import sys
import wave
import time
import logging
import argparse
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

try:
    import pyaudio
    import keyboard
    from groq import Groq
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages: pip install pyaudio keyboard groq python-dotenv")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("audio_transcription")


@dataclass
class AudioConfig:
    """Configuration for audio recording parameters."""
    format: int = pyaudio.paInt16
    channels: int = 1
    rate: int = 16000  # 16kHz is optimal for speech recognition
    chunk: int = 1024
    device_index: Optional[int] = None


class ConfigManager:
    """Manages configuration settings for the application."""
    
    def __init__(self) -> None:
        """Initialize configuration with defaults and environment variables."""
        # Load environment variables
        load_dotenv(override=True)
        
        # API configuration
        self.api_key: Optional[str] = os.getenv("GROQ_API_KEY")
        self.model: str = os.getenv("GROQ_STT", "whisper-large-v3-turbo")
        
        # Recording configuration
        self.audio_config = AudioConfig()
        self.trigger_key: str = "space"
        
        # File management
        self.output_dir: Path = Path(os.getenv("OUTPUT_DIR", os.getcwd()))
        self.default_filename: str = "recorded_audio.wav"
        
        # API request configuration
        self.request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration values and create directories if needed."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate API settings
        if not self.api_key:
            logger.warning("GROQ_API_KEY environment variable not set")
            
        # Validate audio settings
        if self.audio_config.rate < 8000 or self.audio_config.rate > 48000:
            logger.warning(f"Unusual sample rate: {self.audio_config.rate} Hz")
    
    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """Get the full path for the output file."""
        return self.output_dir / (filename or self.default_filename)


class AudioRecorder:
    """Handles audio recording functionality."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize the audio recorder with the given configuration."""
        self.config = config
        self.audio_config = config.audio_config
        self._pyaudio_instance = None
    
    @property
    def pyaudio_instance(self) -> pyaudio.PyAudio:
        """Get or create PyAudio instance."""
        if self._pyaudio_instance is None:
            self._pyaudio_instance = pyaudio.PyAudio()
        return self._pyaudio_instance
    
    def list_devices(self) -> List[Dict[str, Any]]:
        """List available audio input devices."""
        devices = []
        for i in range(self.pyaudio_instance.get_device_count()):
            device_info = self.pyaudio_instance.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': device_info['defaultSampleRate']
                })
        return devices
    
    def calculate_audio_level(self, audio_data: bytes) -> float:
        """Calculate audio level from raw audio data."""
        import struct
        import numpy as np
        
        fmt = f"{len(audio_data)//2}h"  # Format string for unpacking
        data = struct.unpack(fmt, audio_data)
        data_np = np.array(data, dtype=np.float32) / 32768.0  # Normalize to [-1, 1]
        
        # Calculate RMS level
        rms = np.sqrt(np.mean(np.square(data_np)))
        return rms
    
    def display_audio_level(self, level: float, width: int = 40) -> None:
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
    
    @contextmanager
    def open_stream(self):
        """Context manager for audio stream handling."""
        audio = self.pyaudio_instance
        stream = audio.open(
            format=self.audio_config.format,
            channels=self.audio_config.channels,
            rate=self.audio_config.rate,
            input=True,
            input_device_index=self.audio_config.device_index,
            frames_per_buffer=self.audio_config.chunk
        )
        
        try:
            yield stream
        finally:
            stream.stop_stream()
            stream.close()
    
    def record_audio(self, output_filename: Optional[str] = None, 
                   key_to_hold: Optional[str] = None) -> Optional[str]:
        """Record audio while the specified key is held down."""
        key = key_to_hold or self.config.trigger_key
        output_path = self.config.get_output_path(output_filename)
        
        logger.info(f"Hold the {key.upper()} key to start recording...")
        
        # Wait for key press
        while not keyboard.is_pressed(key):
            time.sleep(0.1)
        
        logger.info("Recording started! Keep holding the key...")
        
        # Record while key is held
        frames = []
        try:
            with self.open_stream() as stream:
                while keyboard.is_pressed(key):
                    data = stream.read(self.audio_config.chunk, exception_on_overflow=False)
                    frames.append(data)
                    
                    # Display audio level every few chunks to reduce CPU usage
                    if len(frames) % 2 == 0:  # Every other chunk
                        level = self.calculate_audio_level(data)
                        self.display_audio_level(level)
        except Exception as e:
            logger.error(f"Error during recording: {e}")
            return None
        finally:
            # Move to the next line after the level display
            print()
        
        logger.info("Recording stopped.")
        
        # Save the recorded audio to WAV file
        if frames:
            try:
                with wave.open(str(output_path), 'wb') as wf:
                    wf.setnchannels(self.audio_config.channels)
                    wf.setsampwidth(self.pyaudio_instance.get_sample_size(self.audio_config.format))
                    wf.setframerate(self.audio_config.rate)
                    wf.writeframes(b''.join(frames))
                logger.info(f"Audio saved to {output_path}")
                return str(output_path)
            except Exception as e:
                logger.error(f"Error saving audio file: {e}")
                return None
        else:
            logger.warning("No audio recorded.")
            return None
    
    def close(self) -> None:
        """Close the PyAudio instance and free resources."""
        if self._pyaudio_instance:
            self._pyaudio_instance.terminate()
            self._pyaudio_instance = None


class TranscriptionService:
    """Service for audio transcription using Groq API."""
    
    def __init__(self, config: ConfigManager) -> None:
        """Initialize the transcription service."""
        self.config = config
        self._client = None
    
    @property
    def client(self) -> Groq:
        """Get or create the Groq client."""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError("GROQ_API_KEY not provided or set in environment")
            self._client = Groq(api_key=self.config.api_key)
        return self._client
    
    def _attempt_transcription(self, audio_file_path: str) -> Dict[str, Any]:
        """Single attempt at transcribing audio."""
        logger.info(f"Transcribing audio using model: {self.config.model}")
        
        with open(audio_file_path, "rb") as file:
            response = self.client.audio.transcriptions.create(
                file=file,
                model=self.config.model,
                response_format="json",
                language="en",
                timeout=self.config.request_timeout
            )
        return response
    
    def transcribe(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio using Groq's Whisper API with retry logic."""
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return None
        
        logger.info("Sending audio to Groq API for transcription...")
        
        # Implement retry logic with exponential backoff
        max_retries = self.config.max_retries
        retry_delay = 1  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                response = self._attempt_transcription(audio_file_path)
                return response.text
            except Exception as e:
                logger.warning(f"Transcription attempt {attempt+1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("All transcription attempts failed")
                    return None
        
        return None


class ApplicationUI:
    """User interface management for the application."""
    
    def __init__(self) -> None:
        """Initialize the UI components."""
        self.parser = self._create_argument_parser()
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Record audio and transcribe it using Groq's Whisper API",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            "-o", "--output",
            help="Output filename for the recorded audio",
            default=None
        )
        
        parser.add_argument(
            "-k", "--key",
            help="Key to hold for recording",
            default="space"
        )
        
        parser.add_argument(
            "-d", "--device",
            help="Audio input device index",
            type=int,
            default=None
        )
        
        parser.add_argument(
            "--list-devices",
            action="store_true",
            help="List available audio input devices and exit"
        )
        
        parser.add_argument(
            "-r", "--rate",
            help="Audio sample rate in Hz",
            type=int,
            default=16000
        )
        
        parser.add_argument(
            "--channels",
            help="Number of audio channels",
            type=int,
            default=1
        )
        
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        
        return parser
    
    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        return self.parser.parse_args()
    
    def apply_args_to_config(self, args: argparse.Namespace, config: ConfigManager) -> None:
        """Apply parsed arguments to the configuration."""
        # Update logging level if verbose is enabled
        if args.verbose:
            logger.setLevel(logging.DEBUG)
        
        # Update audio configuration
        if args.device is not None:
            config.audio_config.device_index = args.device
        
        config.audio_config.rate = args.rate
        config.audio_config.channels = args.channels
        
        # Update trigger key
        if args.key:
            config.trigger_key = args.key
    
    def display_devices(self, devices: List[Dict[str, Any]]) -> None:
        """Display a list of available audio input devices."""
        print("\nAvailable Audio Input Devices:")
        print("-------------------------------")
        for device in devices:
            print(f"Index: {device['index']}")
            print(f"Name: {device['name']}")
            print(f"Channels: {device['channels']}")
            print(f"Default Sample Rate: {device['sample_rate']} Hz")
            print("-------------------------------")
    
    def confirm_action(self, prompt: str) -> bool:
        """Ask the user to confirm an action."""
        print(f"{prompt}")
        print("Press Enter to confirm or any other key to cancel:")
        return input().strip() == ""
    
    def display_transcription(self, text: Optional[str]) -> None:
        """Display the transcription result."""
        print("\n=== Transcription Result ===")
        if text:
            print(text)
        else:
            print("Transcription failed or was cancelled.")
        print("============================\n")


def main() -> None:
    """Main function to orchestrate the audio recording and transcription process."""
    # Initialize components
    ui = ApplicationUI()
    args = ui.parse_args()
    config = ConfigManager()
    ui.apply_args_to_config(args, config)
    
    recorder = AudioRecorder(config)
    transcriber = TranscriptionService(config)
    
    try:
        # List devices if requested
        if args.list_devices:
            devices = recorder.list_devices()
            ui.display_devices(devices)
            return
        
        # Record audio
        logger.info("=== Real-time Audio Transcription with Groq API ===")
        logger.info("This script will record audio and transcribe it using Groq's Whisper API.")
        
        # Record audio when the trigger key is held down
        audio_path = recorder.record_audio(args.output)
        
        if audio_path:
            # Confirm transcription
            if ui.confirm_action("Do you want to send the recorded audio file to Groq for transcription?"):
                # Transcribe the recorded audio
                transcription = transcriber.transcribe(audio_path)
                
                # Display the transcription
                ui.display_transcription(transcription)
            else:
                logger.info("Transcription cancelled by user.")
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
    finally:
        # Cleanup resources
        recorder.close()


if __name__ == "__main__":
    main()
