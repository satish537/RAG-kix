from utilservice import *
from fastapi import status
from fastapi.responses import JSONResponse
from langchain_community.llms.ollama import Ollama
from services.langchain.prompts.mom_prompts3 import *
from services.langchain.text_to_query import text_to_query
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

PROMPT_TEMPLATE = """

{question}

"""

PROMPT_TEMPLATE_TRANSCRIPT = """

{question}


Don't add following content in response:
- 'Here is the response...',
- Meeting Transcript,
- Python code

---


TRANSCRIPT:

        {transcript}

"""


query_list = [Executive_Summary, Meeting_Notes, Other_Key_Point, Action_Items_Template]
key_points = ["Executive_Summary", "Meeting_Notes", "Other_Key_Point", "Action_Item"]

data_path = "././data"

async def text_to_summary(filename: str):

    document_loader = Docx2txtLoader(f"{data_path}/{filename}")
    text = document_loader.load()

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE_TRANSCRIPT)

    response = {
        "Executive_Summary" : [],
        "Meeting_Notes" : [],
        "Other_Key_Point" : [],
        "Action_Item" : []
    }

    for i, query in enumerate(query_list):
        if i != 2:
            prompt = prompt_template.format(question=query, transcript=text)
        elif i == 2:
            other_key_point_temp = PromptTemplate.from_template(query)
            other_key_point_prompt = other_key_point_temp.format(executive_summary=response["Executive_Summary"], meeting_notes=response["Meeting_Notes"])
            prompt = prompt_template.format(question=other_key_point_prompt, transcript=text)

        model = Ollama(model="llama3.1")
        model.keep_alive = -1
        
        response_text = await asyncio.to_thread(model.invoke, prompt)

        response[key_points[i]] = response_text

    action_item_extraction_result = text_to_query("llama3.1", response["Action_Item"], Tasks_and_Decisions_Prompt)
    action_item_extraction_result = action_item_extraction_result.body.decode("utf-8")
    task_content, decisions_content = await parse_task_and_decision(action_item_extraction_result)

    response["Task"] = f"{task_content}"
    response["Decisions"] = f"{decisions_content}"

    short_summary = await create_short_summary(response, model)

    final_summary_object = {
        "MOM_SUMMARY": response,
        "SHORT_SUMMARY": short_summary
    }
 

    # return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)
    return final_summary_object


async def parse_task_and_decision(text):
    print(text)
    # Define regex patterns for tasks and decisions
    task_pattern = r'\*\*Task:?\*\*:?(.*?)\*\*Decision:?\*\*:?'
    decision_pattern = r'\*\*Decision:?\*\*:?(.*)'

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


async def create_short_summary(mom_summary: dict, llm_model: str):

    short_summary_prompt = """
    
    I have a Minutes of Meeting (MoM) summary with sections such as Executive Summary, Meeting Notes, Other Key Points, Tasks, and Decisions. Please generate a short and concise summary adhering to the following guidelines:

    - Present the summary as bullet points, avoiding titles and sub-titles.
    - Organize each point clearly using dashes (-).
    - Analyze and consolidate related points from different sections into cohesive sentences under a unified topic. For example, combine details related to overall project discussions into one point.
    - Ensure no repetition of points or sentences.
    - Retain all key information in a condensed format without losing context.
    - Limit the summary to a maximum of 10-12 points.

    The MoM summary is provided below: {mom_summary}

    """

    ss_prompt_template = ChatPromptTemplate.from_template(short_summary_prompt)
    prompt = ss_prompt_template.format(mom_summary=mom_summary)

    # Ensure the LLM model is verified and available
    llm_model = verify_llm(llm_model)
    model = Ollama(model=llm_model, keep_alive=-1)

    # Invoke the model with the formatted prompt and collect the summary_responses
    summary_responses_text = await asyncio.to_thread(model.invoke, prompt)  # If invoke is synchronous

    return summary_responses_text



