# import spacy
# import re
# import json
# import logging
# from collections import defaultdict
# from difflib import SequenceMatcher
# from sklearn.feature_extraction.text import TfidfVectorizer
# from typing import Dict, List
# from docx import Document




# class AdvancedSpeakerMapper:
#     def __init__(self, confidence_threshold: float = 0.7):
#         self.nlp = spacy.load("en_core_web_sm")
#         self.confidence_threshold = confidence_threshold
#         logging.basicConfig(level=logging.INFO)
#         self.logger = logging.getLogger(__name__)

#     class SpeakerProfile:
#         def __init__(self):
#             self.utterances = []
#             self.self_references = []
#             self.mentioned_names = defaultdict(int)
#             self.linguistic_features = {}
#             self.interaction_patterns = defaultdict(int)

#     def extract_linguistic_markers(self, text: str) -> Dict:
#         """Extract detailed linguistic markers from text"""
#         doc = self.nlp(text)
#         return {
#             'pronoun_count': len([token for token in doc if token.pos_ == 'PRON']),
#             'verb_count': len([token for token in doc if token.pos_ == 'VERB']),
#             'sentence_length': len(doc),
#             'entity_count': len(list(doc.ents))
#         }

#     def map_speakers(self, transcript: str, participants: List[str]) -> Dict:
#         """Main mapping function with advanced validation"""
#         speaker_profiles = defaultdict(self.SpeakerProfile)
#         speaker_mapping = {}
#         conversation_flow = []

#         # First pass: Build speaker profiles
#         for line in transcript.split('\n'):
#             if match := re.match(r'Speaker (\d+):(.*)', line.strip()):
#                 speaker_id, utterance = match.groups()
#                 speaker_id = f"Speaker {speaker_id}"

#                 if not utterance.strip():
#                     continue

#                 profile = speaker_profiles[speaker_id]
#                 profile.utterances.append(utterance.strip())

#                 # Extract linguistic features
#                 profile.linguistic_features.update(
#                     self.extract_linguistic_markers(utterance)
#                 )

#                 # Track conversation flow
#                 conversation_flow.append(speaker_id)

#                 # Extract potential self-references
#                 if "this is" in utterance.lower() or "i am" in utterance.lower():
#                     profile.self_references.append(utterance)

#                 # Extract name mentions
#                 for name in participants:
#                     if name.lower() in utterance.lower():
#                         profile.mentioned_names[name] += 1

#         # Second pass: Calculate confidence and map speakers
#         for speaker_id, profile in speaker_profiles.items():
#             confidence_scores = {}

#             for name in participants:
#                 # Calculate confidence based on multiple factors
#                 score = 0.0

#                 # Check self-references
#                 for ref in profile.self_references:
#                     if name.lower() in ref.lower():
#                         score += 0.6  # Adjusted weight

#                 # Check name mentions
#                 score += profile.mentioned_names.get(name, 0) * 0.4  # Adjusted weight

#                 # Check linguistic patterns
#                 ling_score = self.calculate_linguistic_score(profile, name)
#                 score += ling_score * 0.2

#                 confidence_scores[name] = min(1.0, score)

#             # Log confidence scores for debugging
#             self.logger.info(f"Confidence scores for {speaker_id}: {confidence_scores}")

#             # Find best match
#             if confidence_scores:
#                 best_match, confidence = max(confidence_scores.items(), key=lambda x: x[1])

#                 if confidence >= self.confidence_threshold:
#                     speaker_mapping[speaker_id] = {
#                         'mapped_name': best_match,
#                         'confidence': confidence,
#                         'evidence': {
#                             'self_references': profile.self_references,
#                             'name_mentions': dict(profile.mentioned_names),
#                             'linguistic_features': profile.linguistic_features
#                         }
#                     }
#                 else:
#                     speaker_mapping[speaker_id] = {
#                         'mapped_name': speaker_id,
#                         'confidence': confidence,
#                         'evidence': 'Insufficient confidence for mapping'
#                     }
#             else:
#                 speaker_mapping[speaker_id] = {
#                     'mapped_name': speaker_id,
#                     'confidence': 0.0,
#                     'evidence': 'No matching evidence found'
#                 }

#         return speaker_mapping

#     def calculate_linguistic_score(self, profile: SpeakerProfile, name: str) -> float:
#         """Calculate linguistic pattern score"""
#         if not profile.linguistic_features:
#             return 0.0

#         try:
#             # Calculate average of numerical features
#             numerical_features = [
#                 v for v in profile.linguistic_features.values()
#                 if isinstance(v, (int, float ))
#             ]

#             if not numerical_features:
#                 return 0.0

#             avg_value = sum(numerical_features) / len(numerical_features)

#             # Calculate similarity to name
#             similarity = SequenceMatcher(None, name.lower(), str(avg_value)).ratio()
#             return similarity

#         except Exception as e:
#             self.logger.warning(f"Error calculating linguistic score: {e}")
#             return 0.0

# def print_replaced_transcript(transcript: str, mapping: dict):
#     """Print transcript with speaker IDs replaced by mapped names"""
#     print("\n=== Transcript with Mapped Names ===")
#     print("=" * 50 + "\n")

#     for line in transcript.split('\n'):
#         if line.strip():
#             # Check if line starts with Speaker ID
#             if match := re.match(r'Speaker (\d+):(.*)', line.strip()):
#                 speaker_id, content = match.groups()
#                 speaker_key = f"Speaker {speaker_id}"

#                 # Get mapped name from the mapping
#                 mapped_name = mapping[speaker_key]['mapped_name']

#                 # Print replaced line
#                 print(f"{mapped_name}: {content}")
#             else:
#                 print(line)

# def read_docx(file_path):
#     doc = Document(file_path)
#     content = []
#     for paragraph in doc.paragraphs:
#         content.append(paragraph.text)
#     return '\n'.join(content)

# def main(transcript, participants):

#     transcript = read_docx(transcript)
    
#     # Initialize and run mapper
#     mapper = AdvancedSpeakerMapper()
#     mapping = mapper.map_speakers(transcript, participants)

#     # Print mapping results
#     print("\n=== Speaker Mapping Results ===")
#     print("=" * 50)

#     for speaker, data in mapping.items():
#         print(f"{speaker}: {data['mapped_name']} (Confidence: {data['confidence']:.2f})")
#         print(f"  Evidence: {data['evidence']}")
#         print()

#     # Print the replaced transcript
#     print_replaced_transcript(transcript, mapping)

              