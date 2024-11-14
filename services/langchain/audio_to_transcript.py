import os
import contextlib
import wave
import numpy as np
from fastapi import HTTPException, status
from pydub import AudioSegment
import asyncio
from utilservice import delete_document

# Transcribe WAV using the preloaded Whisper model
def transcribe_audio(model, wav_path):
    try:
        result = model.transcribe(wav_path)
        return result["text"], result["segments"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error during transcription: {e}")

# Read WAV file
def read_wave(path):
    try:
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate
    except Exception as e:
        print(f"Error reading WAV file {path}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error reading WAV file: {e}")

# Frame generator for splitting audio data into frames
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield audio[offset:offset + n]
        offset += n

# Energy-based Voice Activity Detection (VAD) for Speaker Diarization
def energy_based_vad(wav_path):
    try:
        audio, sample_rate = read_wave(wav_path)
        frames = list(frame_generator(30, audio, sample_rate))
        energies = [np.sum(np.frombuffer(frame, dtype=np.int16)**2) for frame in frames]

        # Threshold for detecting speaker change
        threshold = np.mean(energies) * 1.5
        segments = []
        current_segment = []

        for i, energy in enumerate(energies):
            if energy > threshold:
                current_segment.append(i)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []

        if current_segment:
            segments.append(current_segment)

        # Assigning Speaker Labels (this is simplistic and alternates between Speaker1 and Speaker2)
        speaker_labels = ['Speaker1', 'Speaker2']
        labeled_segments = [(segment, speaker_labels[i % 2]) for i, segment in enumerate(segments)]

        return labeled_segments
    except Exception as e:
        print(f"Error in energy-based VAD: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in energy-based VAD: {e}")

# Refine speaker segmentation using punctuation marks to decide speaker changes more naturally
def refine_speaker_segmentation(transcription_segments, labeled_segments):
    try:
        refined_segments = []
        current_speaker = labeled_segments[0][1]
        segment_text = ""
        segment_start = transcription_segments[0]['start']

        for i, segment in enumerate(transcription_segments):
            text = segment['text']
            segment_text += text + " "

            # Detect punctuation for potential speaker change
            if any(p in text for p in [".", "?", "!"]) or i == len(transcription_segments) - 1:
                refined_segments.append({
                    "start": segment_start,
                    "end": segment['end'],
                    "text": segment_text.strip(),
                    "speaker": current_speaker
                })
                segment_text = ""
                segment_start = segment['end']

                # Change speaker only at the end of the sentence (when punctuation occurs)
                if any(p in text for p in [".", "?", "!"]):
                    current_speaker = 'Speaker1' if current_speaker == 'Speaker2' else 'Speaker2'

        return refined_segments
    except Exception as e:
        print(f"Error refining speaker segmentation: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error refining speaker segmentation: {e}")

# Format transcription into readable text with speaker labels
def format_transcription(refined_segments):
    try:
        formatted_transcription = []
        for segment in refined_segments:
            formatted_transcription.append(f"{segment['speaker']}: {segment['text']}")
        return "\n".join(formatted_transcription)
    except Exception as e:
        print(f"Error formatting transcription: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error formatting transcription: {e}")

# Main process to handle transcription and diarization
async def process_transcript(model, data_path, audiofile_path, recording_id):
    try:
        # Define the paths for the audio and WAV files
        wav_path = os.path.join(data_path, f"{recording_id}.wav")  # WAV file path based on recording ID

        # Ensure the audio file exists
        if not os.path.exists(audiofile_path):
            raise FileNotFoundError(f"Audio file {audiofile_path} not found.")

        print(f"Audio file found: {audiofile_path}")

        # Convert audio to WAV format
        convert_to_wav(audiofile_path, wav_path)

        print(f"File converted to WAV: {wav_path}")

        # Transcribe the audio using the preloaded model
        transcription_text, transcription_segments = transcribe_audio(model, wav_path)

        print("Audio transcription completed.")

        # Apply energy-based segmentation and speaker refinement
        labeled_segments = energy_based_vad(wav_path)
        refined_segments = refine_speaker_segmentation(transcription_segments, labeled_segments)

        # Format the transcription
        formatted_transcription = format_transcription(refined_segments)

        print("Speaker segmentation and refinement completed.")

        # Clean up the audio and WAV file after processing
        audio_dir, audio_filename = os.path.split(audiofile_path)
        wav_dir, wav_filename = os.path.split(wav_path)

        # Use asyncio to run delete_document asynchronously
        await asyncio.to_thread(delete_document, audio_dir, audio_filename)  # Delete the original audio file
        await asyncio.to_thread(delete_document, wav_dir, wav_filename)  # Delete the generated WAV file

        print(f"Original file deleted: {audiofile_path} and WAV file deleted: {wav_path}")
        return {
            "transcription_text": transcription_text,
            "formatted_transcription": formatted_transcription
        }

    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf_error))

    except HTTPException as http_error:
        print(f"HTTP error: {http_error.detail}")
        raise

    except Exception as e:
        print(f"Error processing transcript: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing transcript")

# Convert MP4 or MP3 to WAV
def convert_to_wav(audiofile, wav_path):
    try:
        print(f"Converting {audiofile} to WAV...")

        # Ensure the file exists before conversion
        if not os.path.exists(audiofile):
            print(f"File {audiofile} does not exist!")
            raise FileNotFoundError(f"File {audiofile} not found.")

        audio = AudioSegment.from_file(audiofile)
        audio = audio.set_channels(1)  # Convert to mono
        audio.export(wav_path, format="wav")
        print(f"WAV file saved at {wav_path}")
    except Exception as e:
        print(f"Error occurred while converting {audiofile}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error occurred while converting {audiofile}: {e}") 
