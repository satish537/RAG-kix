import os
import json
import torch
import torchaudio
import whisperx
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from omegaconf import OmegaConf
import wget
import nltk
from typing import Dict

# Download NLTK data
nltk.download('punkt', quiet=True)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
BATCH_SIZE = 16
WHISPER_MODEL_NAME = "large"
PUNCT_MODEL_LANGS = ["en", "fr", "de", "es", "it", "nl", "pt", "bg", "pl", "cs", "sk", "sl"]

class AudioTranscriber:
    def __init__(self, output_dir):
        print("Loading models...")
        self.device = DEVICE
        self.whisper_model = whisperx.load_model(WHISPER_MODEL_NAME, self.device, compute_type=COMPUTE_TYPE)
        self.diarizer = NeuralDiarizer(cfg=self.create_config(output_dir)).to(self.device)
        self.punct_model = PunctuationModel(model="kredor/punctuate-all")
        print("Models loaded successfully.")

    def create_config(self, output_dir):
        DOMAIN_TYPE = "telephonic"
        CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
        if not os.path.exists(MODEL_CONFIG):
            MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

        config = OmegaConf.load(MODEL_CONFIG)
        config.diarizer.manifest_filepath = "temp_manifest.json"
        config.diarizer.out_dir = output_dir
        config.num_workers = 1
        config.batch_size = 64
        return config


    def process_audio_file(self, audio_path: str, output_dir: str, enable_stemming: bool = True) -> Dict:
        print(f"Processing {audio_path}...")
    # Load and process audio
        if enable_stemming:
        # Load audio and ensure it is in the correct shape
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.dim() == 1:  # Mono audio
                waveform = waveform.unsqueeze(0)  # Add channel dimension
            audio = waveform.numpy()  # Convert to numpy array
        else:
           audio = whisperx.load_audio(audio_path)

    # Check if audio has the correct shape (1, time) or (channels, time)
        if audio.ndim != 2:
            raise ValueError(f"'audio' must be a 2D numpy array (channels, time). Got shape: {audio.shape}")

    # Transcribe
        result = self.whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
        language = result["language"]

    # Align
        model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

    # Diarize
        diarizer_manifest = {
        "audio_filepath": audio_path,
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "num_speakers": None,
        "rttm_filepath": None,
        "uem_filepath": None
        }
        with open("temp_manifest.json", "w") as f:
            json.dump(diarizer_manifest, f)
        diar_hyp = self.diarizer.diarize()

    # Assign speakers
        result = whisperx.assign_word_speakers(diar_hyp, result)

    # Apply punctuation
        words_list = [word.word for word in result["words"]]
        if language in PUNCT_MODEL_LANGS:
            words_list = self.punct_model.predict(words_list)
            for word, (predicted_word, _) in zip(result["words"], words_list):
                word.word = predicted_word

    # Format output
        output = self.format_result(result)

    # Save result
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_result.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Processing completed. Results saved to {output_path}")
        return outputprint(f"Processing {audio_path}...")

    def format_result(self, result: Dict) -> Dict:
        transcript = []
        words = []
        for segment in result["segments"]:
            segment_words = []
            for word in segment["words"]:
                word_data = {
                    "word": word.word,
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "speaker": word.speaker
                }
                segment_words.append(word.word)
                words.append(word_data)
            segment_data = {
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "speaker": segment["speaker"],
                "text": " ".join(segment_words)
            }
            transcript.append(segment_data)

        return {
            "transcript": transcript,
            "words": words
        }
