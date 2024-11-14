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


Don't add following content in response:
- 'Here is the response...',
- Meeting Transcript,
- Python code

---


TRANSCRIPT:

        {transcript}

"""

Main_Prompt = """

You are tasked with generating an Executive Summary and Meeting Notes based on the provided meeting transcript. 

1. **Executive Summary**: 
   - Create a concise executive summary that encapsulates the key points discussed during the meeting. 
   - The summary should be written in a single paragraph and contain between 150-200 words. 
   - Focus solely on the major takeaways, decisions made, and important action items, avoiding any additional details or background information.

2. **Meeting Notes**:
   - Generate detailed meeting notes that are structured with sub-headings based on the main topics discussed during the meeting. 
   - Each sub-heading should correspond to a specific agenda item or discussion point. 
   - Provide a thorough breakdown of the discussion, including key points, participant contributions, decisions made, and any follow-up actions required. 
   - Ensure clarity and organization to facilitate understanding and reference for future meetings.

Please analyze the meeting transcript provided and produce the Executive Summary and Meeting Notes accordingly.

"""

async def generate_summary(filename: str, data_path: str):

    document_loader = Docx2txtLoader(f"{data_path}/{filename}")
    text = document_loader.load()

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(question=Main_Prompt, transcript=text)

    model = Ollama(model="qwen2.5:7b")
    model.keep_alive = -1
    
    response_text = await asyncio.to_thread(model.invoke, prompt)
 
    print(response_text)
    # return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)
    return response_text





