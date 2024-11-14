from utilservice import *
from pprint import pprint
from time import time
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.ollama import Ollama
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from services.langchain.text_to_query import text_to_query
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from services.langchain.embedding import get_embedding_function
from services.langchain.momSummary.text_to_mom import text_to_summary
import json
from time import sleep

from services.langchain.singleton import OllamaSingleton

# ollama_model = OllamaSingleton.get_instance()
ollama_model = Ollama(model="llama3.2:3b", keep_alive=-1)

CHROMA_DB_PATH = "././chroma/langchain"

# Templates for each section

executive_summary_template = """
Generate a concise, professional executive summary based on the provided transcript. The summary should be one cohesive paragraph. Should capture all key points discussed, decisions made, and any next steps or actions agreed upon. Be clear, concise, and written in professional business English. Use formal language and avoid colloquialisms or casual expressions. Be specific and avoid vague statements. Not include unnecessary details or placeholders. Reflect the actual points discussed in the meeting. Be written in a formal, business-like tone. Do not hallucinate the timelines. Please generate text without any markdown headers or hashes.

General guidelines for the outputs:
Ensure all content is based solely on the provided transcript. Do not fabricate content or include placeholders.. If no clear conclusions or decisions were reached, briefly state that. If the transcript does not cover specific discussions or topics for any section, state "No specific discussions or future actions identified in this transcript" for that section.

Transcript:

{transcript}
"""

# Template for generating detailed meeting notes from a transcript
meeting_notes_template = """
Provide detailed meeting notes based on the transcript. The notes should clearly outline all the topics discussed, specific feedback or comments given, and any comparisons made. Also include additional suggestions, discussions, future plans, and priorities. Structure the notes by topic, and ensure that each section is clearly organized using dash (-) points for clarity. If a section or topic has a title, wrap the title with single asterisks (*). Do not include sections like "Meeting Summary," "Next Steps," "Action Items," "Conclusion," or "Follow-Up Actions." Ensure the headings reflect the actual topics discussed in the meeting, and focus on summarizing the key points without adding unnecessary headers. Please generate text without any markdown headers or hashes.

The meeting notes should follow this format:

*Topic 1* \\n
- Describe about Topic 1 \\n,
- Describe about Topic 1 \\n,

*Topic 2* \\n
- Describe about Topic 2 \\n,
- Describe about Topic 2 \\n,

(Continue listing all topics discussed in the meeting)

General guidelines for the outputs:
- Only include topics based on the transcript; do not fabricate content or include placeholders.
- If no clear conclusions or decisions were reached, briefly state that.
- If the transcript does not cover specific discussions or topics for any section, state: "No specific discussions or future actions identified in this transcript."

Ensure the meeting notes are written in professional business English, with clear, concise formatting, avoiding any casual language.

Transcript:

{transcript}

"""


action_items_template = """
Based on the provided transcript, generate a list of only the **most important and high-priority action items** discussed during the meeting. Focus on key decisions and tasks that have a significant impact on the project or require urgent attention. Avoid including minor tasks, repetitive actions, or general suggestions. Summarize the actions concisely and clearly, while keeping the language professional and specific. If no action item was explicitly mentioned, do not fabricate content or include placeholders. Avoid assigning timelines unless they were clearly discussed in the transcript. Please generate text without any markdown headers or hashes.

Format the action items as follows:

- Action Item 1: [describe the key task/decision],

(Continue listing only the major and high-priority action items)

General guidelines for the outputs:
Ensure all content is based solely on the provided transcript. Do not fabricate content or include placeholders. If no clear conclusions or decisions were reached, briefly state that. If the transcript does not cover specific discussions or topics for any section, state "No specific discussions or future actions identified in this transcript" for that section

Transcript:

{transcript}

"""

query_templates = [executive_summary_template, meeting_notes_template, action_items_template]
query_keys = ["Executive_Summary", "Meeting_Notes", "Action_Item"]

def get_executive_summary_final_prompt(combined_transcript):
    return f"""
    Generate a concise, professional executive summary based on the provided transcript. The summary should be only one consize and cohesive paragraph. Should capture key points discussed, decisions made, and any next steps or actions agreed upon. Be clear, concise, and written in professional business English. Use formal language and avoid colloquialisms or casual expressions. Be specific and avoid vague statements. Not include unnecessary details or placeholders. Reflect the actual points discussed in the meeting. Do not hallucinate the timelines. Please generate text without any markdown headers or hashes.


    Transcript:
    {combined_transcript}
    """

def get_meeting_notes_final_prompt(combined_transcript):
    return f"""

    Template:

    *Introduction*
    - Brief introduction of all participants and their roles.

    *Topic Name*
    - Detailed discussion points related to the topic:
      - Key points
      - Any challenges or issues identified

    *Another Topic Name*
    - Detailed discussion points related to another topic:
      - Key points
      - Any challenges or issues identified

    ... (continue for all topics discussed in the meeting)

    (Note: Ensure each topic is described without any sections or sub-sections, maintaining a continuous flow of information as seen from the transcript. Follow the above specified template and make a note that it cover all the topics in the given transcript. Make a note that the topic name should not be action items. Leave the topics which are mentioned like no specific discussion mentioned in transcript. Under each topic the discussion points, key points, challenges should be like points no need to use any subheadings for those points and the format should not be a paragraph.)
    Replace Topic Name with the actual topics discussed during the meeting as mentioned in the transcript. This format ensures professional and concise notes that adhere to business standards. Please generate text without any markdown headers or hashes.

    Transcript:
    {combined_transcript}

    """

def get_action_items_final_prompt(combined_transcript):
    return f"""
    Based on the provided transcript, generate a list of only the **most important and high-priority action items** discussed during the meeting. Focus on key decisions and tasks that have a significant impact on the project or require urgent attention. Avoid including minor tasks, repetitive actions, or general suggestions. Summarize the actions concisely and clearly, while keeping the language professional and specific. If no action item was explicitly mentioned, do not fabricate content or include placeholders. Avoid assigning timelines unless they were clearly discussed in the transcript. Please generate text without any markdown headers or hashes.

    Format the action items as follows:
    - Action Item 1: [describe the key task/decision],

    (Continue listing only the major and high-priority action items)


    General guidelines for the outputs:
    - Ensure all content is based solely on the provided transcript.
    - Do not fabricate content or include placeholders.
    - If no specific discussions or decisions were reached for a topic, state: "No specific action items identified in this transcript."

    Transcript:

    {combined_transcript}
    """

async def generate_prompt_mom(llm_model, data_path, filename, recording_id):
    start_time = time()
    print(f"MOM summary request start for recordingID: {recording_id} Time: {start_time}")

    # Load the transcript file based on file type
    _, file_extension = os.path.splitext(filename)
    if file_extension == '.txt':
        document_loader = TextLoader(f"{data_path}/{filename}")
    elif file_extension in ['.docx', '.doc']:
        document_loader = Docx2txtLoader(f"{data_path}/{filename}")
    elif file_extension == '.pdf':
        document_loader = PyPDFLoader(f"{data_path}/{filename}")
    elif file_extension == '.csv':
        document_loader = CSVLoader(f"{data_path}/{filename}")
    else:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File format not supported")

    documents = await asyncio.to_thread(document_loader.load)
    print(f"Loaded document content for recordingID: {recording_id} Time: {time() - start_time:.2f}")

    # Split the document into chunks for better context handling
    chunk_size = 6000
    chunk_overlap = 1000
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Handle longer transcripts
    docs = text_splitter.split_documents(documents)

    print(f"Document split into {len(docs)} chunks with size: {chunk_size}, overlap: {chunk_overlap} for recordingID: {recording_id}")

    summary_responses = {key: [] for key in query_keys}

    for i, query in enumerate(query_templates):
        # print(f"Processing section {query_keys[i]} for recordingID: {recording_id} Time: {time() - start_time:.2f}", "\n")
        for chunk_num, doc in enumerate(docs):
            context_text = doc.page_content
            prompt = query.format(transcript=context_text)
            summary_responses_text = await asyncio.to_thread(ollama_model.invoke, prompt)

            # Append chunk responses to corresponding section
            summary_responses[query_keys[i]].append(summary_responses_text.strip())

    final_summary = {}
    for key in query_keys:
        combined_transcript = "\n".join(summary_responses[key])

        # Generate the final prompt based on section
        if key == "Executive_Summary":
            prompt = get_executive_summary_final_prompt(combined_transcript)
        elif key == "Meeting_Notes":
            prompt = get_meeting_notes_final_prompt(combined_transcript)
        elif key == "Action_Item":
            prompt = get_action_items_final_prompt(combined_transcript)

        sleep(5)
        final_response = await asyncio.to_thread(ollama_model.invoke, prompt)
        final_summary[key] = final_response.strip()

        print(f"Finalized summary for {key} for recordingID: {recording_id} Time: {time() - start_time:.2f}", "\n")

    # Combine all sections into one text block to generate a short summary
    full_summary_text = "\n\n".join(final_summary.values())
    print(f"Combined full summary for recordingID: {recording_id} Time: {time() - start_time:.2f}", "\n")

    # Generate the short summary
    short_summary_prompt = get_short_summary_final_prompt(full_summary_text)
    sleep(10)
    short_summary = await asyncio.to_thread(ollama_model.invoke, short_summary_prompt)
    print(f"Short summary generated for recordingID: {recording_id} Time: {time() - start_time:.2f}", "\n")

    final_summary_object = {
        "MOM_SUMMARY": final_summary,
        "SHORT_SUMMARY": short_summary.strip()
    }

    execution_time = f"{time() - start_time:.2f}"
    print(f"MOM Summary generated successfully for recordingID: {recording_id} in {execution_time} seconds.", "\n")
    await cleanup(data_path, filename, recording_id)
    return final_summary_object

# Final prompt for short summary
def get_short_summary_final_prompt(full_summary):
    return f"""
    Based on the following full meeting summary, generate a concise short summary. Limit the summary to 10-12 bullet points.
    Ensure each point is unique, concise, and avoids repetition.

    Full Meeting Summary:
    {full_summary}
    """

# Clean up and delete temporary files
async def cleanup(data_path, filename, recording_id):
    await asyncio.to_thread(delete_document, data_path, filename)
