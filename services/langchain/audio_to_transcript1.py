import ffmpeg
import torch
import torchaudio
import whisperx
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import asyncio
import warnings
import os
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend.*")

# Function to detect available GPUs
def get_available_gpus():
    gpu_count = torch.cuda.device_count()
    # Use only GPUs 0, 1, 2, and 3 if there are at least 4 GPUs available
    if gpu_count >= 4:
        return {"device": "cuda", "device_index": [0, 1, 2, 3]}
    elif gpu_count > 1:
        return {"device": "cuda", "device_index": list(range(gpu_count))}
    elif gpu_count == 1:
        return {"device": "cuda", "device_index": 0}
    else:
        return {"device": "cpu", "device_index": 0}  # Fallback to CPU if no GPUs

# Function to extract audio from video using ffmpeg
async def extract_audio_from_video(video_path, output_audio_path="extracted_audio.wav"):
    try:
        ffmpeg.input(video_path).output(output_audio_path).run(quiet=True, overwrite_output=True)
        # print(f"Audio extracted to: {output_audio_path}")
        return output_audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

# Function to perform clustering based on voice embeddings
async def hybrid_clustering(embeddings, similarity_threshold=0.5):
    if len(embeddings) < 2:
        # print("Not enough embeddings for clustering. Skipping clustering.")
        return np.zeros(len(embeddings))  # Return a single speaker label for all segments

    # Step 1: DBSCAN for clustering with cosine similarity
    dbscan = DBSCAN(eps=similarity_threshold, min_samples=2, metric="cosine").fit(embeddings)
    dbscan_labels = dbscan.labels_

    # Fallback to Agglomerative Clustering if DBSCAN doesn't find enough clusters
    if len(set(dbscan_labels)) <= 1:
        # print("DBSCAN failed to find enough clusters. Falling back to Agglomerative Clustering.")
        agglomerative = AgglomerativeClustering(linkage='average')
        agglomerative_labels = agglomerative.fit_predict(embeddings)
        return agglomerative_labels
    else:
        return dbscan_labels

# Function to perform speaker diarization based on the embeddings
async def perform_speaker_diarization(audio_path, segments, sr, timing_threshold=0.75, similarity_threshold=0.5):
    wav, _ = torchaudio.load(audio_path)
    wav = wav.mean(dim=0).numpy()  # Convert to mono
    wav = preprocess_wav(wav, source_sr=sr)

    encoder = VoiceEncoder()
    # print("Loaded the voice encoder model")

    embeddings = []
    timing_diffs = []
    previous_end = 0

    for segment in segments:
        start, end = segment["start"], segment["end"]
        segment_wav = wav[int(start * sr):int(end * sr)]
        embedding = encoder.embed_utterance(segment_wav)
        embeddings.append(embedding)

        timing_diffs.append(start - previous_end)
        previous_end = end

    embeddings = np.array(embeddings)

    # Ensure embeddings is a 2D array, even if only one embedding exists
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)  # Single sample case

    # print(f"Shape of embeddings: {embeddings.shape}")

    # **Check if we have more than 1 embedding before clustering**
    if len(embeddings) < 2:
        # print("Only one speaker detected, skipping clustering.")
        # Assign all segments the same speaker label (e.g., Speaker 0 or Speaker 1)
        for segment in segments:
            segment["speaker"] = 0  # Assign a single speaker label
        return segments

    # Continue with clustering if there are multiple embeddings
    speaker_labels = await hybrid_clustering(embeddings, similarity_threshold=similarity_threshold)

    num_speakers = len(set(speaker_labels)) - (1 if -1 in speaker_labels else 0)
    # print(f"Estimated number of speakers: {num_speakers}")

    for i, segment in enumerate(segments):
        if i > 0 and timing_diffs[i] > timing_threshold:
            speaker_labels[i] = (speaker_labels[i - 1] + 1) % num_speakers
        segment["speaker"] = speaker_labels[i]

    return segments

# Transcribe audio using WhisperX with VAD
async def transcribe_audio_with_whisperx(audio_path, whisperx_model):
    transcription = whisperx_model.transcribe(audio_path)
    
    # Check if transcription contains segments
    if not transcription.get("segments"):
        # print("No transcription segments found. Skipping diarization.")
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)
    aligned_transcription = whisperx.align(transcription["segments"], model_a, metadata, audio_path, device=device)
    return aligned_transcription

# Extract word-level segments from WhisperX transcription
async def extract_word_level_segments(aligned_transcription):
    word_segments = []
    for segment in aligned_transcription["segments"]:
        word_segments.extend(segment["words"])  # Collect all word-level information
    return word_segments

# Refine speaker segmentation based on WhisperX transcription and Resemblyzer outputs
async def refine_speaker_segmentation(transcription_segments, speaker_segments):
    refined_segments = []
    min_len = min(len(transcription_segments), len(speaker_segments))

    for i in range(min_len):
        try:
            text = transcription_segments[i]['text']
            speaker = speaker_segments[i]['speaker']
            refined_segments.append({
                "start": float(transcription_segments[i]['start']),
                "end": float(transcription_segments[i]['end']),
                "text": text,
                "speaker": int(speaker)
            })
        except KeyError as e:
            print(f"KeyError while refining speaker segmentation: {e}")
            continue

    return refined_segments

# Function to format transcription with speaker labels as plain text
async def format_plain_text_transcription(refined_segments):
    formatted_transcription = []
    current_speaker = None

    for segment in refined_segments:
        if segment["speaker"] != current_speaker:
            current_speaker = segment["speaker"]
            formatted_transcription.append(f"Speaker {current_speaker}: {segment['text']}\n")
        else:
            formatted_transcription.append(f"Speaker {current_speaker}: {segment['text']}\n")

    return "\n".join(formatted_transcription)

# Function to format transcription with word-level details as JSON
async def format_json_word_transcription(refined_segments, aligned_transcription, delay=0.01):
    word_segments = await extract_word_level_segments(aligned_transcription)
    word_diarized_segments = []

    previous_end_time = 0.0
    for word in word_segments:
        word_start = word.get("start", previous_end_time + delay)
        word_end = word.get("end", word_start + delay)

        previous_end_time = word_end

        for segment in refined_segments:
            if segment["start"] <= word_start <= segment["end"]:
                word_diarized_segments.append({
                    "word": word["word"],
                    "start": float(word_start),
                    "end": float(word_end),
                    "confidence": float(word.get("confidence", 1.0)),
                    "speaker": int(segment["speaker"]),
                    "speaker_confidence": 1.0
                })

    return word_diarized_segments

# Main transcription process function
async def process_transcript_test(whisperx_model, data_path, audio_path, recording_id):
    try:
        # Step 1: Transcribe audio using WhisperX
        aligned_transcription = await transcribe_audio_with_whisperx(audio_path, whisperx_model)

        if aligned_transcription is None:
            print("Skipping transcription as no segments were found.")
            return None

        # Step 2: Perform speaker diarization using Resemblyzer
        wav_info = torchaudio.info(audio_path)
        sr = wav_info.sample_rate
        speaker_segments = await perform_speaker_diarization(audio_path, aligned_transcription["segments"], sr)

        # Step 3: Refine speaker segmentation
        refined_segments = await refine_speaker_segmentation(aligned_transcription["segments"], speaker_segments)

        # Step 4: Generate plain text transcription with speaker labels
        plain_text_transcription = await format_plain_text_transcription(refined_segments)

        # Step 5: Generate JSON with word-level aligned transcription
        json_word_transcription = await format_json_word_transcription(refined_segments, aligned_transcription)

        # Clean up audio files after processing (optional)
        await asyncio.to_thread(os.remove, audio_path)

        # Step 6: Return both plain text transcription and word-level JSON transcription
        return {
            "formatted_transcription": plain_text_transcription,
            "aligned_segments": json_word_transcription
        }

    except Exception as e:
        print(f"Error processing transcript: {e}")
        raise Exception(f"Error processing transcript: {e}")
