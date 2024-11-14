from typing import List
import os


# Function to replace speaker IDs with participant names
def replace_speaker_ids(transcript: str, participants: List[str]) -> str:
    lines = transcript.split('\n')
    updated_lines = []

    for line in lines:
        if line.startswith("Speaker"):
            # Extract speaker ID
            parts = line.split(': ', 1)
            speaker_id = int(parts[0].split()[1])  # Get the speaker ID number
            # Check if the speaker ID is within bounds and map to participant name or return speaker ID
            if speaker_id < len(participants):
                participant_name = participants[speaker_id]
                updated_line = f"{participant_name}: {parts[1]}"
            else:
                updated_line = f"Speaker {speaker_id}: {parts[1]}"  # Return the speaker ID if no participant found

            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)  # Keep any non-speaker lines unchanged

    return '\n'.join(updated_lines)
