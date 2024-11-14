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
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *
from pprint import pprint
from time import sleep, time
import re

# Path to Chroma database
DB_PATH = "././chroma/langchain"

# Template for creating prompts with question and transcript
PROMPT_TEMPLATE = """
{question}

---

Transcript:

        {transcript}
"""

# Template for generating an executive summary from a meeting transcript
Executive_Summary = """
Please provide an executive summary of the following meeting transcript. 
The summary should include the key points discussed, decisions made, and any future actions or plans mentioned. 
Every key point have only one or two sentence. Don't generate long point. 
All the point start with dash (-).
The summary should be concise and follow this format:

- Key point or topic 1, \\n
- Key point or topic 2, \\n
- Key point or topic 3, \\n

NOTE : Don't given extra information in response, just return only executive summary key point of summary like format. \n if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript". 
"""

# Template for generating detailed meeting notes from a transcript
Meeting_Notes = """
Please provide detailed description of meeting notes based on the following meeting transcript. 
The notes should include the main topics discussed, specific feedback given, comparisons made, and any next steps or action items identified. 
Give me each section is detailed and well-organized, using dash (-) points for clarity. if have point have a title then wrap title with single astrict (*).
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
 ...

NOTE : if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript".
"""

Other_Key_Point = """
Please provide a summary of the other key points discussed in the following meeting transcript. 
The summary should include additional important points that were not covered in the main discussion, focusing on suggestions, priorities, future plans, and enhancements. 
Every key point have only one or two sentence. Don't generate long point. 
All the point start with dash (-).
The summary should be concise and follow this format:

- Additional Key Point 1, \\n
- Additional Key Point 2, \\n
- Additional Key Point 3 \\n

NOTE : Don't given extra information in response, just return only other key point of summary like format. \n if transcript is not able generate summary of above topic then only return "No specific decisions or future actions discussed in this transcript". 
"""

Action_Item = """
Please provide an Action item of the following meeting.
Action item cover Task, Decision, Plan, etc.
Every key point have only one or two sentence. Don't generate long point.
All the point start with dash (-).
The Action item should be well-organized and follow this format:

- Action item 1, \\n
- Action item 2, \\n
- Action item 3  \\n
"""

query_list = [Executive_Summary, Meeting_Notes, Other_Key_Point, Action_Item]
key_point = ["Executive_Summary", "Meeting_Notes", "Other_Key_Point", "Action_Item"]

async def generate_prompt_test(llm_model, data_path, filename, recording_id):
    print("file_to_mom called for", recording_id)
    start_time = time()

    # Extract the file extension to determine the document type
    _, file_extension = os.path.splitext(filename)

    # Choose the appropriate loader based on the file extension
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
            print("File Format is Not Supported")
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="File Format is Not Supported")

    # Load the documents using the selected loader
    documents = document_loader.load()
    # Split the document into smaller chunks to handle large files
    text_splitter = CharacterTextSplitter(chunk_size=2500, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    # Initialize Chroma for document embedding and retrieval
    db = Chroma(
        persist_directory=f"{DB_PATH}/{recording_id}", embedding_function=get_embedding_function()
    )
    # Ensure proper permissions for the persistence directory
    sleep(1)
    persist_directory = f"{DB_PATH}/{recording_id}"
    os.chmod(persist_directory, 0o755)

    embedding_function = get_embedding_function()

    # Add documents to the database and persist them
    db.add_documents(docs)
    db.persist()
    # Reload the database for subsequent operations
    db = Chroma(persist_directory=f"{DB_PATH}/{recording_id}", embedding_function=embedding_function)

    # Get the number of results/documents stored in the database
    no_of_result = db._collection.count()

    # Initialize response structure to store results
    response = {
        "Executive_Summary": [],
        "Meeting_Notes": [],
        "Other_Key_Point": [],
        "Action_Item": [],
        "Task": [],
        "Decisions": []
    }

    # Process each query from the query list to generate responses
    for i, query in enumerate(query_list):
        results = db.similarity_search_with_score(query, k=no_of_result)
        results_list = partition_list(results, 25)  # Batch results into groups of 25
        batched_prompts = []

        for result in results_list:
            # Combine the contents of the documents in the result set
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result])

            # Format the prompt using the template and the provided query and transcript
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(question=query, transcript=context_text)
            batched_prompts.append(prompt)

        # Combine the batched prompts into a single string separated by a special delimiter
        combined_prompt = "\n\n===\n\n".join(batched_prompts)

        # Ensure the LLM model is verified and available
        llm_model = verify_llm(llm_model)
        model = Ollama(model=llm_model, keep_alive=-1)

        # Synchronously invoke the model with the combined prompt
        combined_response = model.invoke(combined_prompt)

        # Split the combined response back into individual responses using the same delimiter
        individual_responses = combined_response.split("\n\n===\n\n")

        # Append the individual responses to the appropriate list in the response dictionary
        response[key_point[i]].extend(individual_responses)

    RESPONSE_TEMPLATE = """
    *Aparna provided an overview of the Jamie platform and its key features:*
        - Recording and transcribing meetings
        - Identifying and naming speakers
        - Generating meeting minutes with executive summary, notes, decisions, tasks, and participants
        - Sharing meeting minutes through a shareable link
    *Feedback on the Jamie platform:*
        -Positives: Well-structured and high-quality output
        -Negatives: Meeting minute credits limit (5 free credits per month, each credit for 30 mins meeting), Billing model not ideal for frequent usage (e.g. $24 for 120 mins/15 credits)
    """

    # Template and instructions for creating a comprehensive summary
    summary_prompt = f"""
    
    You have a list of summaries derived from individual meeting transcripts. 
    Your task is to create a comprehensive and concise final summary that captures all the essential points discussed across all the meetings. 
    The summary point should be very detailed.
    Ensure to remove any duplicate points and present the final summary in a well-organized format.

    Instructions:

        1) Review each summary carefully.
        2) Extract all distinct points mentioned in the summaries.
        3) Remove any duplicate points to ensure each point is unique.
        4) Combine these distinct points into a cohesive final summary.
        5) Format the final summary as a Python list, with each point as a separate list item.
        6) Ensure the final summary is professional and clearly organized.
        7) Give me each section is detailed and well-organized, using dash (-) points for clarity. if have point have a title then wrap title with single astrict (*). 

    For better understanding here is template for generate response and response must be like this : {RESPONSE_TEMPLATE}
    This response template is just one example for understand formatting. Don't add any content in response from response template.
    
    """

    # Template and instructions for separating tasks and decisions from action items
    Task_Decision_prompt = """
    
    You have a list of Action Item from the Meeting Transcript.
    Please separate Task and Decision from the above Action Item points.one Action Point going to only one section(Task, Decision).
     
    Instructions:

        1) Review each Action point carefully.
        2) Extract all distinct points mentioned in the transcript.
        3) Remove any duplicate points to ensure each point is unique.
        4) Ensure the final points is professional and clearly organized.
        5) Give me each section is detailed and well-organized, using dash (-) points for clarity. 
        6) Formate of Response is like :
            **Task**:
                - [Task 1]
                - [Task 2]
                - [Task 3]
            
            **Decision**:
                - [Decision 1]
                - [Decision 2]
                - [Decision 3]
                
    """
    # Process the gathered responses and format them into final outputs
    for i in range(3):
        if no_of_result <= 2:
            # Combine responses if the number of results is small
            combined_response = "".join(response[key_point[i]])
            response[key_point[i]] = combined_response
        else:
            # Generate a final summary by querying a summarization model
            final_summary = text_to_query("mistral", response[key_point[i]], summary_prompt)
            response[key_point[i]] = final_summary.body.decode("utf-8")

    # Separate tasks and decisions if there are multiple action items
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

    # Clean up: Delete the processed document and the temporary directory
    delete_document(data_path, filename)
    delete_directory(directory_path=f"{DB_PATH}/{recording_id}")

    final_summary_object = {
        "MOM_SUMMARY": response,
    }

    print("---------------------------------------------------------------------------------------------------------------------------------------")
    print("The MOM Summary for Recording ID", recording_id, "was successfully generated in", "%.2f" % (time() - start_time), "seconds.")
    print("---------------------------------------------------------------------------------------------------------------------------------------")
    return final_summary_object
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

# Function to extract task and decision content from an Action Item using regular expressions
def extract_task_and_decision_content(text):
    # Define regex patterns for tasks and decisions
    task_pattern = r'\*\*Tasks?:?\*\*(.*?)\*\*Decisions?:?\*\*'
    decision_pattern = r'\*\*Decisions?:?\*\*(.*)'

    # Extract the content between **Task** and **Decision**
    task_content = ""
    task_match = re.search(task_pattern, text, re.DOTALL)
    if task_match:
        task_content = task_match.group(1).strip()

    # Extract the content after **Decision**
    decision_content = ""
    decision_match = re.search(decision_pattern, text, re.DOTALL)
    if decision_match:
        decision_content = decision_match.group(1).strip()

    return task_content, decision_content

async def callback_mom(llm_model, DATA_PATH, filename, recording_id):
    print("callback_mom called for ", recording_id)
    start_time = time()
    response = asyncio.run(generate_prompt_test(llm_model, DATA_PATH, filename, recording_id))

    my_json = response.body.decode('utf8').replace("'", '"')

    # Load the JSON to a Python list & dump it back out as formatted JSON
    data = json.loads(my_json)
    returnn = {
        "recordingId": recording_id,
        "response": data,
        "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
    }
