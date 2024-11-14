import lamini, os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
from fastapi.responses import JSONResponse
from fastapi import status, HTTPException
from utilservice import *



RESPONSE_TEMPLATE = """
   **Meeting Report:**
* **Summary:** Key topics & outcomes (bullet points).
* **Notes:** Detailed notes by topic with examples.
* **Other Points:** Significant points not covered elsewhere.
* **Decisions:**
    * Identified (bullet points)
    * Potential (quotes from transcript)
    * Unclear (need for further review)
* **Tasks & Action Items:** Owner, deadline, description (bullet points).
    NOTE: all of the list separated by star(*) points. if you not found about any topic related data in transcript then return None in that topic.
"""


PROMPT_TEMPLATE = """
 **Meeting Transcript:**

{transcript}

**Meeting Report:**
 * Give insights on the topics and key outcomes discussed in the meeting as a sentence and use bullet points for clarity
 * Organize by all the topics and key outcomes discussed in the meeting along with technical aspects / topics with detailed summarized description for each topic and outcome using clear headings. Ensure comprehensive coverage without unnecessary repetition. Use bullet points for clarity.
 * Please provide a summary of other key points from the following meeting transcript.This summary should include detailed information on all significant points that were not covered in the meeting notes, tasks, and decisions.Focus on capturing the main points and avoid minor or granular details.  
 * Briefly summarize the major decisions made during this meeting. Please focus on key takeaways and action items, and avoid including minor details or discussions. Include clear ownership (who is responsible) and target completion dates for each decision using bullet points.
 * Maintain a professional tone and focus on essential information.
    For better understanding here is template for generate response and response must be like this : {template}
    """



def text_to_m2mquery(text: str, llm_model: str):

    try:
        prompt_template = PromptTemplate.from_template(
            template=PROMPT_TEMPLATE
            )

        llm_model = verify_llm(llm_model)
        llm = Ollama(model=llm_model)

        chain = prompt_template | llm | StrOutputParser()
        
        response_text = chain.invoke({"transcript": text, "template": RESPONSE_TEMPLATE})


        return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)

    except Exception as error:
        
        handel_exception(error)
