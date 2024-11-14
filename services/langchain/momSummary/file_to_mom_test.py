import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader, UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import CharacterTextSplitter
from services.langchain.text_to_query import text_to_query
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from services.langchain.embedding import get_embedding_function
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *
from time import sleep, time
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor

DB_PATH = "././chroma/langchain"

PROMPT_TEMPLATE = """
{question}
---
Transcript:
{transcript}
"""

Executive_Summary = """
Please provide an executive summary of the following meeting transcript.
The summary should include the key points discussed, decisions made, and any future actions or plans mentioned.
Every key point should have only one or two sentences. Don't generate long points.
All points should start with a dash (-).
The summary should be concise and follow this format:
- Key point or topic 1, \\n
- Key point or topic 2, \\n
- Key point or topic 3, \\n
NOTE: Don't give extra information in response, just return only the executive summary key points in the format above. If the transcript does not have the required information, return "No specific decisions or future actions discussed in this transcript".
"""

Meeting_Notes = """
Please provide detailed meeting notes based on the following meeting transcript.
The notes should include the main topics discussed, specific feedback given, comparisons made, and any next steps or action items identified.
Give each section in detail and well-organized, using dash (-) points for clarity. If a point has a title, wrap the title with a single asterisk (*).
The notes should be well-organized and follow this format:
*Topic 1* \\n
- Details about Topic 1 \\n,
- Details about Topic 1 \\n,
- Details about Topic 1 \\n,
*Topic 2* \\n
- Details about Topic 2 \\n,
- Details about Topic 2 \\n,
*Next Steps* \\n,
- Details about Next Steps \\n,
- Details about Next Steps \\n,
NOTE: If the transcript does not have the required information, return "No specific decisions or future actions discussed in this transcript".
"""

Other_Key_Point = """
Please provide a summary of the other key points discussed in the following meeting transcript.
The summary should include additional important points that were not covered in the main discussion, focusing on suggestions, priorities, future plans, and enhancements.
Every key point should have only one or two sentences. Don't generate long points.
All points should start with a dash (-).
The summary should be concise and follow this format:
- Additional Key Point 1, \\n
- Additional Key Point 2, \\n
- Additional Key Point 3 \\n
NOTE: Don't give extra information in response, just return only the other key points in the format above. If the transcript does not have the required information, return "No specific decisions or future actions discussed in this transcript".
"""

Action_Item = """
Please provide the action items from the following meeting.
Action items should cover tasks, decisions, plans, etc.
Every key point should have only one or two sentences. Don't generate long points.
All points should start with a dash (-).
The action items should be well-organized and follow this format:
- Action item 1, \\n
- Action item 2, \\n
- Action item 3  \\n
"""

query_list = [Executive_Summary, Meeting_Notes, Other_Key_Point, Action_Item]
key_point = ["Executive_Summary", "Meeting_Notes", "Other_Key_Point", "Action_Item"]

def generate_prompt(llm_model, data_path, filename, recording_id):
    start_time = time()
    _, file_extension = os.path.splitext(filename)
    match file_extension:
        case '.txt':
            document_loader = TextLoader(f"{data_path}/{filename}")
        case '.docx' | '.doc':
            document_loader = Docx2txtLoader(f"{data_path}/{filename}")
        case '.pdf':
            document_loader = PyPDFLoader(f"{data_path}/{filename}")
        case '.csv':
            document_loader = CSVLoader(f"{data_path}/{filename}")
        case _:
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File Format is Not Supported")

    documents = document_loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    db = Chroma(persist_directory=f"{DB_PATH}/{recording_id}", embedding_function=get_embedding_function())
    sleep(1)
    persist_directory = f"{DB_PATH}/{recording_id}"
    os.chmod(persist_directory, 0o755)
    embedding_function = get_embedding_function()
    db.add_documents(docs)
    db.persist()
    db = Chroma(persist_directory=f"{DB_PATH}/{recording_id}", embedding_function=embedding_function)
    no_of_result = db._collection.count()
    response = {
        "Executive_Summary": [],
        "Meeting_Notes": [],
        "Other_Key_Point": [],
        "Action_Item": [],
        "Task": [],
        "Decisions": []
    }

    def process_query(i, query):
        results = db.similarity_search_with_score(query, k=no_of_result)
        results_list = split_list(results, 25)
        for result in results_list:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(question=query, transcript=context_text)
            model = Ollama(model=llm_model, keep_alive=-1)
            response_text = model.invoke(prompt)
            response[key_point[i]].append(response_text)

    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, query in enumerate(query_list):
            executor.submit(process_query, i, query)

    RESPONSE_TEMPLATE = """
    *Aparna provided an overview of the Jamie platform and its key features:*
        - Recording and transcribing meetings
        - Identifying and naming speakers
        - Generating meeting minutes with executive summary, notes, decisions, tasks, and participants
        - Sharing meeting minutes through a shareable link
    *Feedback on the Jamie platform:*
        - Positives: Well-structured and high-quality output
        - Negatives: Meeting minute credits limit (5 free credits per month, each credit for 30 mins meeting), Billing model not ideal for frequent usage (e.g. $24 for 120 mins/15 credits)
    """

    summary_prompt = f"""
    You have a list of summaries derived from individual meeting transcripts.
    Your task is to create a comprehensive and concise final summary that captures all the essential points discussed across all the meetings.
    Ensure to remove any duplicate points and present the final summary in a well-organized format.
    Instructions:
        1) Review each summary carefully.
        2) Extract all distinct points mentioned in the summaries.
        3) Remove any duplicate points to ensure each point is unique.
        4) Combine these distinct points into a cohesive final summary.
        5) Format the final summary as a Python list, with each point as a separate list item.
        6) Ensure the final summary is professional and clearly organized.
        7) Give each section in detail and well-organized, using dash (-) points for clarity. If a point has a title, wrap the title with a single asterisk (*).
    This response template is just one example for understanding formatting. Don't add any content in response from the response template.
    """

    Task_Decision_prompt = """
    You have a list of Action Items from the Meeting Transcript.
    Please separate Tasks and Decisions from the above Action Items points. One Action Point should go to only one section (Task, Decision).
    Instructions:
        1) Review each Action point carefully.
        2) Extract all distinct points mentioned in the transcript.
        3) Remove any duplicate points to ensure each point is unique.
        4) Ensure the final points are professional and clearly organized.
        5) Give each section in detail and well-organized, using dash (-) points for clarity.
        6) Format the response as follows:
            **Task**:
                - [Task 1]
                - [Task 2]
                - [Task 3]
            **Decision**:
                - [Decision 1]
                - [Decision 2]
                - [Decision 3]
    """

    for i in range(3):
        if no_of_result <= 2:
            combined_response = "".join(response[key_point[i]])
            response[key_point[i]] = combined_response
        else:
            final_summary = text_to_query("mistral", response[key_point[i]], summary_prompt)
            response[key_point[i]] = final_summary.body.decode("utf-8")

    if no_of_result >= 2:
        Action_item_extract_res = text_to_query("mistral", response["Action_Item"], Task_Decision_prompt)
        Action_item_extract_res = Action_item_extract_res.body.decode("utf-8")
        Task_content, Decisions_content = extract_task_and_decision_content(Action_item_extract_res)
        response["Task"] = Task_content
        response["Decisions"] = Decisions_content
        response.pop("Action_Item")
    else:
        response.pop("Task")
        response.pop("Decisions")

    delete_document(data_path, filename)
    delete_directory(directory_path=f"{DB_PATH}/{recording_id}")
    print("**")
    print(time()-start_time)
    return JSONResponse(content=response, status_code=status.HTTP_200_OK)

def split_list(lst, max_size):
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

def extract_task_and_decision_content(text):
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
