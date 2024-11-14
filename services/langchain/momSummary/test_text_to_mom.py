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

* Summarize key topics and outcomes (bullet points).
* Organize detailed notes by major topics with specific examples.
* Include any significant points not covered elsewhere.
* Identify clear decisions and list them (bullet points).
* If unclear, flag potential decisions with transcript quotes.
* If no decisions are clear, state the need for further review.
* List tasks & action items with owners, deadlines, and descriptions (bullet points).
* Maintain a professional tone and focus on essential information.
    For better understanding here is template for generate response and response must be like this : {template}
    """



def text_to_m2mquery(text: str, llm_model: str):

    prompt_template = PromptTemplate.from_template(
        template=PROMPT_TEMPLATE
        )

    llm_model = verify_llm(llm_model)
    llm = Ollama(model=llm_model)

    chain = prompt_template | llm | StrOutputParser()
    
    response_text = chain.invoke({"transcript": text, "template": RESPONSE_TEMPLATE})


    return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)
