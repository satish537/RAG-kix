import os, asyncio, ast, json, requests
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import CharacterTextSplitter
from services.langchain.text_to_query import text_to_query
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from services.langchain.embedding import get_embedding_function
from services.langchain.loadDocuments.transcriptDoc import transcriptToChunks
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *
from pprint import pprint
from time import sleep, time
from services.langchain.singleton import OllamaSingleton

ollama_model = OllamaSingleton.get_instance()


# Path to Chroma database
DB_PATH = "./././chroma/langchain"


# Template for creating prompts with question and transcript
PROMPT_TEMPLATE = """

{question}

---

Transcript:

        {transcript}

"""


# Templates for each section

executive_summary_template = """
Generate a concise, professional executive summary based on the provided transcript. The summary should be one cohesive paragraph. Should capture all key points discussed, decisions made, and any next steps or actions agreed upon. Be clear, concise, and written in professional business English. Use formal language and avoid colloquialisms or casual expressions. Be specific and avoid vague statements. Not include unnecessary details or placeholders. Reflect the actual points discussed in the meeting. Be written in a formal, business-like tone. Do not hallucinate the timelines.

General guidelines for the outputs:
Ensure all content is based solely on the provided transcript. Do not fabricate content or include placeholders.. If no clear conclusions or decisions were reached, briefly state that. If the transcript does not cover specific discussions or topics for any section, state "No specific discussions or future actions identified in this transcript" for that section.

Transcript:

{transcript}
"""

# Template for generating detailed meeting notes from a transcript
meeting_notes_template = """
Provide detailed meeting notes based on the transcript. The notes should clearly outline all the topics discussed, specific feedback or comments given, and any comparisons made. Also include additional suggestions, discussions, future plans, and priorities. Structure the notes by topic, and ensure that each section is clearly organized using dash (-) points for clarity. If a section or topic has a title, wrap the title with single asterisks (*). Do not include sections like "Meeting Summary," "Next Steps," "Action Items," "Conclusion," or "Follow-Up Actions." Ensure the headings reflect the actual topics discussed in the meeting, and focus on summarizing the key points without adding unnecessary headers.

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
Based on the provided transcript, generate a list of only the **most important and high-priority action items** discussed during the meeting. Focus on key decisions and tasks that have a significant impact on the project or require urgent attention. Avoid including minor tasks, repetitive actions, or general suggestions. Summarize the actions concisely and clearly, while keeping the language professional and specific. If no action item was explicitly mentioned, do not fabricate content or include placeholders. Avoid assigning timelines unless they were clearly discussed in the transcript.

Format the action items as follows:

- Action Item 1: [describe the key task/decision],

(Continue listing only the major and high-priority action items)

General guidelines for the outputs:
Ensure all content is based solely on the provided transcript. Do not fabricate content or include placeholders. If no clear conclusions or decisions were reached, briefly state that. If the transcript does not cover specific discussions or topics for any section, state "No specific discussions or future actions identified in this transcript" for that section

Transcript:

{transcript}

"""



query_list = [executive_summary_template, meeting_notes_template, action_items_template]
key_point = ["Executive_Summary", "Meeting_Notes", "Action_Item"]


async def generate_mom_summary_large(llm_model, filename, projectID, recording_id):

    # Reload the database for subsequent operations
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=f"{DB_PATH}/{projectID}", embedding_function=embedding_function
    )

    # Initialize response structure to store results
    response = {
        "Executive_Summary" : [],
        "Meeting_Notes" : [],
        "Action_Item" : [],
    }

    # Process each query from the query list to generate responses
    for i, query in enumerate(query_list):

        results = db.similarity_search_with_score(query, k=100, filter={"recordingID": recording_id})
        results_list = partition_list(results, 12)

        for result in results_list:

            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result])

            prompt_template = ChatPromptTemplate.from_template(query)
            prompt = prompt_template.format(transcript=context_text)

            response_text = ollama_model.invoke(prompt)

            response[key_point[i]].append(response_text)

    final_summary = {}

    for key in key_point:
        combined_transcript = "\n".join(response[key])

        if key == "Executive_Summary":
            prompt = get_executive_summary_final_prompt(combined_transcript)
        elif key == "Meeting_Notes":
            prompt = get_meeting_notes_final_prompt(combined_transcript)
        elif key == "Action_Item":
            prompt = get_action_items_final_prompt(response["Action_Item"])


        final_response = await asyncio.to_thread(ollama_model.invoke, prompt)
        final_summary[key] = final_response.strip()

    full_summary_text = "\n\n".join(final_summary.values())

    short_summary_prompt = get_short_summary_final_prompt(full_summary_text)

    short_summary = await asyncio.to_thread(ollama_model.invoke, short_summary_prompt)

    final_summary_object = {
        "MOM_SUMMARY": final_summary,
        "SHORT_SUMMARY": short_summary.strip()
    }

    return final_summary_object


# Final prompt for short summary
def get_short_summary_final_prompt(full_summary):
    return f"""
    Based on the following full meeting summary, generate a concise short summary. Limit the summary to 10-12 bullet points.
    Ensure each point is unique, concise, and avoids repetition.

    Full Meeting Summary:
    {full_summary}
    """

def get_executive_summary_final_prompt(combined_transcript):
    return f"""
    Generate a concise, professional executive summary based on the provided transcript. The summary should be only one consize and cohesive paragraph. Should capture key points discussed, decisions made, and any next steps or actions agreed upon. Be clear, concise, and written in professional business English. Use formal language and avoid colloquialisms or casual expressions. Be specific and avoid vague statements. Not include unnecessary details or placeholders. Reflect the actual points discussed in the meeting. Do not hallucinate the timelines.

    Please generate text without any markdown headers or hashes.

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
    Replace Topic Name with the actual topics discussed during the meeting as mentioned in the transcript. This format ensures professional and concise notes that adhere to business standards.

    Please generate text without any markdown headers or hashes.

    Transcript:
    {combined_transcript}

    """

def get_action_items_final_prompt(combined_transcript):
    return f"""

    You are an AI tasked with consolidating a list of action items from various inputs. 
    Your objective is to take the following action items, which may contain repetition or minor variations, and return them as a single, cohesive list. 
    Each item in the list should be unique, combining similar content where necessary and ensuring clarity and conciseness within one line. 
    Maintain the original format of the action items while eliminating duplicates and enhancing the overall readability.
    Please generate text without any markdown headers or hashes. 

    **Output Format:**
    - Action Item 1: [describe the key task/decision]
    - Action Item 2: [describe the key task/decision]
    - Action Item 3: [describe the key task/decision]
    - ... 

    Please process the action items provided and return the final consolidated list in the specified format.

    {combined_transcript}
    """




# Function to split a list into smaller sublists
def partition_list(lst, max_size):
    num_elements = len(lst)
    # Calculate the size of each group and the number of extra elements
    full_size = num_elements // max_size
    remainder = num_elements % max_size
    
    result = []
    start = 0
    
    for i in range(full_size):
        result.append(lst[start:start + max_size])
        start += max_size
    
    # Handle the remainder part
    if remainder:
        result.append(lst[start:])
    
    return result












