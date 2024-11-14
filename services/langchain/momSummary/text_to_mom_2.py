import lamini, os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
from fastapi.responses import JSONResponse
from fastapi import status, HTTPException
from utilservice import *



RESPONSE_TEMPLATE = """
   **Meeting Report:**

**Executive Summary:**

*  ... (Summarize main topics and key outcomes from the transcript)

**Meeting Notes:**

* **Topic 1:**
    * Key point 1 with a specific example from the transcript.
    * Key point 2 with another specific example.
    * ... (Continue for other key points under Topic 1)
* **Topic 2:**
    * ... (Follow the same structure as Topic 1 for other major topics)

**Other Key Points:**

*  ... (List any additional significant points not covered elsewhere)
*  ... (Mention any discussions that might have led to implicit decisions, if applicable)

**Decisions:**

* **Identified Decisions:**
    * Decision 1 (clearly stated based on the transcript)
    * Decision 2 (clearly stated)
    * ... (Continue for other identified decisions)
* **Potential Decisions (if applicable):**
    * **Potential Decision:** [Quote a relevant section from the transcript] 
    * ... (Continue for other potential decisions)
* **Unclear Decisions:** Based on the transcript, it's unclear what specific decisions were made. Further review or clarification might be needed. (Use this section if no clear or potential decisions are identified)

**Tasks & Action Items:**

* **[Owner Name]:** [Action Item description] by [Deadline] (from transcript)
* **[Team/Role]:** [Action Item description] by [Deadline] (from transcript)
* ... (Continue for other tasks and action items)
    NOTE: all of the list separated by star(*) points. if you not found about any topic related data in transcript then return None in that topic.
"""


PROMPT_TEMPLATE = """
**Meeting Transcript:**

{transcript}  **Meeting Report:**

**Please generate a comprehensive meeting report based on the provided transcript. The report should include the following sections:**

* **Executive Summary:**
    * Briefly summarize the main topics discussed in the meeting and any key outcomes.
    * Use bullet points for clarity.
* **Meeting Notes:**
    * Organize by major topics discussed, using clear headings for each section.
    * Include detailed points and specific examples mentioned under each topic.
    * Ensure comprehensive coverage without unnecessary repetition.
* **Other Key Points:**
    * Capture any additional significant points not already covered elsewhere in the report.
    * Include any discussions that might have led to implicit decisions.
    * Use bullet points for clarity.
* **Decisions:**
    * **Identified Decisions:** List all decisions made during the meeting that are clearly stated in the transcript.
    * Ensure clear and unambiguous wording.
    * **Potential Decisions (if applicable):**
        * If there are sections in the transcript that suggest potential decisions, include them here with a qualifier.
        * Use the format: "**Potential Decision:**  [Quote a relevant section from the transcript]".
    * If there's no evidence of clear decisions in the transcript, you can state: "Based on the transcript, it's unclear what specific decisions were made. Further review or clarification might be needed."
* **Tasks & Action Items:**
    * List all action items assigned during the meeting.
    * Include details like:
        * Who is responsible for the task (owner)
        * Expected deadline for completion
        * Briefly describe the action item
    * Use bullet points for easy identification.

**Note:**

* Maintain a professional tone throughout the report.
* Focus on capturing essential information and avoiding irrelevant details.
* If the transcript quality is poor or the discussion is unclear, consider mentioning these limitations in the report.
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
