import os
import re
import asyncio
import json
import cv2
import difflib
from fastapi import HTTPException, status
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import CharacterTextSplitter
from services.langchain.text_to_query import text_to_query
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from services.langchain.embedding import get_embedding_function
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from PIL import Image
from utilservice import delete_document, delete_directory
from utilservice import *
from fastapi.responses import JSONResponse
from time import time

# Path to Chroma database
CHROMA_DB_PATH = "././chroma/langchain"

# Define prompt templates
CORE_PROMPT_TEMPLATE = """
{question}

---

Transcript:

{transcript}
"""

executive_summary_template = """
Please provide an executive summary of the following meeting transcript.
The summary should include the key points discussed, decisions made, and any future actions or plans mentioned.
For each key point, include a reference to the specific lines in the transcript from which the point was generated, in "(Line x-y)" where x and y are the starting and ending lines of the transcript. The x and y values should not exceed the number of transcript lines given.
The summary should be concise and follow this format:

- Key point or topic 1 (Line x-y) \\n
- Key point or topic 2 (Line x-y)  \\n
- Key point or topic 3 (Line x-y) \\n

NOTE: If the transcript does not have specific decisions or actions, mention "No specific decisions or future actions discussed."
"""

meeting_notes_template = """
Please provide detailed meeting notes based on the following meeting transcript.
Include the main topics discussed, feedback given, comparisons made, and any next steps or action items identified.
For each point, include a reference to the specific lines in the transcript from which the point was generated, in this format: "(Line x-y)" where x and y are the starting and ending lines of the transcript. The x and y values should not exceed the number of transcript lines given.
Organize the notes using dash (-) points for clarity. Use the following format:

*Topic 1* \\n
- Details about Topic 1 (Line x-y) \\n
- Details about Topic 1 (Line x-y) \\n

*Topic 2* \\n
- Details about Topic 2 (Line x-y)  \\n
- Details about Topic 2 (Line x-y)  \\n

*Next Steps* \\n
- Details about Next Steps (Line x-y) \\n
- Details about Next Steps (Line x-y) \\n
"""

other_key_point_template = """
Please provide a summary of the other key points discussed in the following meeting transcript.
The summary should include additional important points that were not covered in the main discussion, focusing on suggestions, comments, or concerns raised.
Every key point should have only one or two sentences. Don't generate long points.
All the points start with dash (-).
The summary should be concise and follow this format:

- Additional Key Point 1 (Line x-y) \\n
- Additional Key Point 2 (Line x-y) \\n
- Additional Key Point 3 (Line x-y) \\n

NOTE: For each point, include a reference to the specific lines in the transcript from which the point was generated, in this format: "(Line x-y)" where x and y are the starting and ending lines of the transcript.  The x and y values should not exceed the number of transcript lines given.
"""

action_items_template = """
Please provide Action items for the following meeting.
Action items cover Tasks, Decisions, Plans, etc.
Every key point should have only one or two sentences. Don't generate long points.
All points start with dash (-).
The Action items should be well-organized and follow this format:

- Action item 1 (Line x-y) \\n
- Action item 2 (Line x-y) \\n
- Action item 3 (Line x-y) \\n

NOTE: For each action item, include a reference to the specific lines in the transcript from which the point was generated, in this format: "(Line x-y)" where x and y are the starting and ending lines of the transcript. The x and y values should not exceed the number of transcript lines given.
"""

query_templates = [executive_summary_template, meeting_notes_template, other_key_point_template, action_items_template]
query_keys = ["Executive_Summary", "Meeting_Notes", "Other_Key_Point", "Action_Item"]

async def generate_prompt_mom_test(llm_model, data_path, filename, recording_id):
    try:
        start_time = time()

        file_path = os.path.join(data_path, filename)
        _, file_extension = os.path.splitext(filename)

        # Choose the appropriate loader based on the file extension
        if file_extension == '.txt':
            document_loader = TextLoader(os.path.join(data_path, filename))
        elif file_extension in ['.docx', '.doc']:
            document_loader = Docx2txtLoader(os.path.join(data_path, filename))
        elif file_extension == '.pdf':
            document_loader = PyPDFLoader(os.path.join(data_path, filename))
        elif file_extension == '.csv':
            document_loader = CSVLoader(os.path.join(data_path, filename))
        else:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File Format is Not Supported")

        documents = await asyncio.to_thread(document_loader.load)

        # Split the document into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
        docs = text_splitter.split_documents(documents)

        current_line = 0
        transcript_lines = []
        for doc in docs:
            lines = doc.page_content.split('\n')
            doc.metadata['line_start'] = current_line + 1
            doc.metadata['line_end'] = current_line + len(lines)
            current_line += len(lines)
            transcript_lines.extend(lines)  # Collecting all lines in transcript_lines

        # Initialize Chroma for document embedding and retrieval
        db = Chroma(
            persist_directory=os.path.join(CHROMA_DB_PATH, recording_id),
            embedding_function=get_embedding_function()
        )

        await asyncio.to_thread(db.add_documents, docs)
        await asyncio.to_thread(db.persist)

        result_count = await asyncio.to_thread(db._collection.count)

        summary_responses = {
            "Executive_Summary": [],
            "Meeting_Notes": [],
            "Other_Key_Point": [],
            "Action_Item": [],
            "Task": [],
            "Decisions": []
        }

        for i, query in enumerate(query_templates):
            results = await asyncio.to_thread(db.similarity_search_with_score, query, result_count)
            results_list = chunk_list1(results, 25)

            for result in results_list:
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result])
                line_references = []

                for doc, _score in result:
                    line_start = doc.metadata.get('line_start', 'unknown')
                    line_end = doc.metadata.get('line_end', 'unknown')
                    line_references.append(f"(Line {line_start}-{line_end})")

                prompt_template = ChatPromptTemplate.from_template(CORE_PROMPT_TEMPLATE)
                prompt = prompt_template.format(question=query, transcript=context_text)

                llm_model = verify_llm(llm_model)
                model = Ollama(model=llm_model, keep_alive=-1)

                summary_responses_text = await asyncio.to_thread(model.invoke, prompt)
                summary_responses_with_refs = f"{summary_responses_text.strip()} {' '.join(line_references)}"
                summary_responses[query_keys[i]].append(summary_responses_with_refs)

        # Optional: Further processing of summaries (e.g., combining or refining)
        summary_responses_template = """
        *Aparna provided an overview of the Jamie platform and its key features:*
        - Recording and transcribing meetings
        - Identifying and naming speakers
        - Generating meeting minutes with executive summary, notes, decisions, tasks, and participants
        - Sharing meeting minutes through a shareable link
        *Feedback on the Jamie platform:*
        - Positives: Well-structured and high-quality output
        - Negatives: Meeting minute credits limit (5 free credits per month, each credit for 30 mins meeting), Billing model not ideal
        """

        final_summary_prompt = f"""
        You have a list of summaries derived from individual meeting transcripts. Your task is to create a comprehensive and concise final summary that captures all the essential points discussed across all topics. Ensure to remove any duplicate points and  present the final summary in a well-organized format.

        Instructions:
        1) Review each summary carefully.
        2) Extract all distinct points mentioned in the summaries.
        3) Remove any duplicate points to ensure each point is unique.
        4) Combine these distinct points into a cohesive final summary.
        5) For each summary point, include the corresponding line numbers from the transcript in this format: "(Line x-y)", where x and y are the starting and ending lines in the transcript. The starting and ending line should not exceed the total lines of transcript and the starting and ending line can't be a blank line and the speaker line.
        6) Ensure the final summary is professional and clearly organized.
        7) Format the final summary as a Python list, with each point as a separate list item.
        8) Ensure the final summary is professional and clearly organized, using dash (-) points for clarity.
        """

        tasks_and_decisions_prompt = """
        You have a list of Action Items from the Meeting Transcript. Please separate Tasks and Decisions from the above Action Items points. One Action Point goes to only one section (Task, Decision).

        Instructions:
        1) Review each Action point carefully.
        2) Extract all distinct points mentioned in the transcript.
        3) Remove any duplicate points to ensure each point is unique.
        4) For each task and decision, include the corresponding line numbers from the transcript in this format: "(Line x-y)", where x and y are the starting and ending lines in the transcript. The starting and ending line should not exceed the total lines of transcript and the starting and ending line can't be a blank line and the speaker line.
        5) Give me each section is detailed and well-organized, using dash (-) points for clarity.
        6) Format of summary_responses is like:
        **Task**:
            - [Task 1] (Line x-y)
            - [Task 2] (Line x-y)
            - [Task 3] (Line x-y)

        **Decision**:
            - [Decision 1] (Line x-y)
            - [Decision 2] (Line x-y)
            - [Decision 3] (Line x-y)
        """

        for i in range(3):
            if result_count <= 2:
                combined_summary_responses = "".join(summary_responses[query_keys[i]])
                summary_responses[query_keys[i]] = combined_summary_responses
            else:
                print(f"Final Summary Prompt for {query_keys[i]} and {recording_id} in {time() - start_time}")
                final_summary = text_to_query("mistral", summary_responses[query_keys[i]], final_summary_prompt)
                summary_responses[query_keys[i]] = final_summary.body.decode("utf-8")

        if result_count >= 2:
            action_item_extraction_result = text_to_query("mistral", summary_responses["Action_Item"], tasks_and_decisions_prompt)
            action_item_extraction_result = action_item_extraction_result.body.decode("utf-8")
            task_content, decisions_content = parse_task_and_decision1(action_item_extraction_result)
            if task_content != "" and decisions_content != "":
                summary_responses["Task"] = task_content
                summary_responses["Decisions"] = decisions_content
                summary_responses.pop("Action_Item")
            elif task_content == "" and decisions_content == "":
                summary_responses.pop("Task")
                summary_responses.pop("Decisions")
        else:
            summary_responses.pop("Task")
            summary_responses.pop("Decisions")

        return summary_responses, transcript_lines

    except Exception as e:
        print(f"Error generating MoM for {recording_id}: {e}")
        raise

def load_timestamps(timestamps_file):
    _, timestamps_extension = os.path.splitext(timestamps_file)
    if timestamps_extension.lower() == '.json':
        with open(timestamps_file, 'r', encoding='utf-8') as file:
            timestamps_data = json.load(file)
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Timestamp file format is not supported."
        )
    return timestamps_data

def find_timestamp_range(transcript_lines, timestamps, start_line, end_line):
    start_time = float('inf')
    end_time = 0
    transcript_text = " ".join(transcript_lines[start_line - 1:end_line]).lower().split()
    best_start_time = None
    best_end_time = None

    for i in range(len(timestamps) - len(transcript_text)):
        current_text = " ".join([t['word'].lower() for t in timestamps[i:i + len(transcript_text)]])
        ratio = difflib.SequenceMatcher(None, " ".join(transcript_text), current_text).ratio()

        if ratio > 0.8:
            if best_start_time is None:
                best_start_time = timestamps[i]['start']
            best_end_time = timestamps[i + len(transcript_text) - 1]['end']

            if ratio > 0.95:
                break

    if best_start_time is None or best_end_time is None:
        raise ValueError(f"Timestamp range could not be found for lines {start_line}-{end_line}")

    return best_start_time, best_end_time

def save_highlight_videos(highlight_points, transcript_lines, timestamps, video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    for i, (start_line, end_line) in enumerate(highlight_points):
        try:
            start_time, end_time = find_timestamp_range(transcript_lines, timestamps, start_line, end_line)
            output_video_path = os.path.join(output_folder, f"highlight_{i}.mp4")
            ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_video_path)
            print(f"Saved video highlight {i} as 'highlight_{i}.mp4'")

            # Capture the starting screenshot
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
            ret, frame_start = cap.read()
            if ret:
                start_image_path = os.path.join(output_folder, f"highlight_{i}_start.jpg")
                Image.fromarray(frame_start).save(start_image_path)
                print(f"Saved start screenshot for highlight {i} as 'highlight_{i}_start.jpg'")

            # Capture the ending screenshot
            cap.set(cv2.CAP_PROP_POS_MSEC, end_time * 1000)
            ret, frame_end = cap.read()
            if ret:
                end_image_path = os.path.join(output_folder, f"highlight_{i}_end.jpg")
                Image.fromarray(frame_end).save(end_image_path)
                print(f"Saved end screenshot for highlight {i} as 'highlight_{i}_end.jpg'")

        except ValueError as e:
            print(f"Could not generate highlight {i}: {str(e)}")

    cap.release()

async def process_mom_and_generate_videos(llm_model, data_path, video_filename, transcript_filename, timestamps_filename, recording_id):
    transcript_path = os.path.join(data_path, transcript_filename)
    timestamps_data = load_timestamps(os.path.join(data_path, timestamps_filename))

    summary_responses, transcript_lines = await generate_prompt_mom_test(llm_model, data_path, transcript_filename, recording_id)
    highlight_points = []

    for key, response in summary_responses.items():
        lines = response.splitlines()
        for line in lines:
            if "(Line " in line:
                # Match both ranges and single lines
                line_matches = re.findall(r"\(Line (\d+)(?:-(\d+))?\)", line)
                for match in line_matches:
                    if len(match) == 2 and match[1]:  # Line range (e.g., (Line 3-5))
                        start_line = int(match[0])
                        end_line = int(match[1])
                    else:  # Single line (e.g., (Line 3))
                        start_line = int(match[0])
                        end_line = start_line  # Single line means start and end are the same
                    highlight_points.append((start_line, end_line))

    output_folder = "highlight_videos"
    save_highlight_videos(highlight_points, transcript_lines, timestamps_data, os.path.join(data_path, video_filename), output_folder)
    # Delete all the files after processing
    await asyncio.to_thread(delete_document, data_path, transcript_filename)
    await asyncio.to_thread(delete_document, data_path, video_filename)
    await asyncio.to_thread(delete_document, data_path, timestamps_filename)
    await asyncio.to_thread(delete_directory, f"{CHROMA_DB_PATH}/{recording_id}")

    return JSONResponse(status_code=status.HTTP_200_OK, content={"MOM_SUMMARY": summary_responses, "output_folder": output_folder})

# Function to split a list into smaller sublists
def chunk_list1(lst, max_size):
    num_elements = len(lst)
    full_size = num_elements // max_size
    remainder = num_elements % max_size

    result = []
    start = 0

    for i in range(full_size):
        result.append(lst[start:start + max_size])
        start += max_size

    if remainder:
        result.append(lst[start:])

    return result

# Function to extract task and decision content from a text using regular expressions
def parse_task_and_decision1(text):
    task_pattern = r'\*\*Task:?\*\*(.*?)\*\*Decision:?\*\*'
    decision_pattern = r'\*\*Decision:?\*\*(.*)'
    task_content = ""
    task_match = re.search(task_pattern, text, re.DOTALL)
    if task_match:
        task_content = task_match.group(1).strip()

    decision_content = ""
    decision_match = re.search(decision_pattern, text, re.DOTALL)
    if decision_match:
        decision_content = decision_match.group(1).strip()

    return task_content, decision_content
