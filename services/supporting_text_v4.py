import os, time, re
import numpy as np
from utilservice import *
from torch import threshold
import asyncio, math, request
from typing import List, Dict
from collections import defaultdict
from scipy.spatial.distance import cosine
from fastapi import status, HTTPException
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from services.embedding import get_embedding_function
from services.extractingResponse import get_matching_strings



CHROMA_PATH = "./chroma/vectorDB"
ollamaModel = Ollama(model="llama3.1", keep_alive=-1)


SUPPORTING_TEXT_PROMPT = """

You are tasked with extracting relevant quotes from a transcript based on a given title and description. Please follow these guidelines to ensure accurate and contextually appropriate extraction:

- Identify sentences in the transcript that directly relate to the themes, concepts, or keywords presented in the title and description.
- Ensure that the extracted quotes maintain the original wording from the transcript without alteration.
- While extracting quotes/points don't remove grammatical mistakes. I want 100 percent of the same sentence that is in the transcript.
- Focus on quotes that provide insight, support, or clarification regarding the title and description.
- Review the context of each quote to confirm its relevance and coherence with the title and description.
- Ensure that the quotes do not exceed one or two sentences in length for conciseness. 
- Present the final list of quotes in a clear and organized manner, ensuring that it is easy to read and reference.
- Response should be only list without heading or title. Each point start with "-" or "*".

Input:

===============================================================================================================================

1. Title: {theme}

2. Description: {description}

3. Transcript: {context}

"""


async def build_chroma_filter(projectID=None, questionId=None, participantId=None, metadata=None):
    conditions = []

    # Add scalar fields
    if projectID:
        conditions.append({"projectId": projectID})
    if questionId:
        conditions.append({"questionId": questionId})
    if participantId:
        conditions.append({"participantId": participantId})

    # Add metadata fields (list or scalar)
    if metadata:
        for key, values in metadata.items():
            if isinstance(values, list):
                if len(values) == 1:
                    conditions.append({key: values[0]})
                elif len(values) > 1:
                    or_block = [{"%s" % key: v} for v in values]
                    conditions.append({"$or": or_block})
            else:
                conditions.append({key: values})

    # Return flat if only one condition, else wrap in $and
    if len(conditions) == 1:
        return conditions[0]
    elif len(conditions) > 1:
        return {"$and": conditions}
    else:
        return {}  # no filters

async def extract_context_from_vector_db(projectID: str, questionId, participantId, metadata):
    """
    Extract context from the vector database based on projectID, questionId, and participantId.
    """
    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    filter_obj = await build_chroma_filter(
        projectID=projectID,
        questionId=questionId,
        participantId=participantId,
        metadata=metadata
    )

    results = db.get(include=["documents", "metadatas"], where=filter_obj)
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"projectId '{projectID}' does not have any matching data"
        )

    # Extract documents and metadata
    documents = results["documents"]
    metadatas = results["metadatas"]

    if len(documents) != len(metadatas):
        raise ValueError("Mismatch between documents and metadata!")

    return results

async def generate_supporting_text(context_text: str, theme, description):
    max_attempts = 5  # Maximum retries
    attempt = 0
    aggregated_points = []  # List to hold all points from all attempts

    while attempt < max_attempts:
        attempt += 1
        # print(f"Attempt {attempt}...for supporting text")
        
        prompt_template = ChatPromptTemplate.from_template(SUPPORTING_TEXT_PROMPT)
        final_prompt = prompt_template.format(context=context_text, theme=theme, description=description)
        response_text = ollamaModel.invoke(final_prompt)
        supporting_text_list = await extract_clean_points(response_text)
        # supporting_text_list = await get_matching_strings(context_text, supporting_text_list)
        aggregated_points.extend(supporting_text_list)

        if len(supporting_text_list) >= 3:
            break

    if len(supporting_text_list) < 3:
        supporting_text_list = aggregated_points

    supporting_text_list = list(set(supporting_text_list))
    # supporting_text_list = [element for element in supporting_text_list if element and element.strip()]

    return supporting_text_list 


async def generate_quotes_details(projectID: str, questionId, participantId, theme, description, metadata, kValue, percentage: float = 50):
    
    results = extract_context_from_vector_db(projectID, questionId, participantId, metadata)

    def group_results_as_tuples(results):
        grouped = defaultdict(list)

        # Group by recording_id
        for doc, meta in zip(results["documents"], results["metadatas"]):
            recording_id = meta.get("recording_id", "unknown")
            grouped[recording_id].append((doc, meta))  # tuple, not dict

        return list(grouped.values())  # Only the groups as a list of lists

    grp_result_list = group_results_as_tuples(results)
    print("stored result in ", len(grp_result_list), "groups")
    all_supporting_texts = []

    for group in grp_result_list:
        """Process a group of results and generate supporting text"""
        # Join the documents in the group
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in group])
        
        # Generate supporting text
        supporting_text_list = await generate_supporting_text(cleaned_context, theme, description)
        
        # Clean and validate supporting texts
        cleaned_texts = []
        for text in supporting_text_list:
            if not text.strip():  # Skip empty or whitespace-only texts
                continue
                
            # Remove quotes, whitespace, commas, and trailing dots
            cleaned_text = re.sub(r"[,.]$", "", text.strip().strip('"\''))
            
            # Handle **text**: pattern
            header_pattern = r"^\*\*(.*?)\*\*\s*:"
            header_match = re.search(header_pattern, cleaned_text)
            if header_match:
                header = header_match.group(1)
                content = cleaned_text[cleaned_text.find(':') + 1:].strip()
                if content:
                    cleaned_text = f"{content} ({header})"
                else:
                    continue
            
            if cleaned_text.strip():  # Only add non-empty texts
                cleaned_texts.append(cleaned_text)
        
            # Get complete context for sorting
            full_context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            validated_list = sort_strings_by_text_occurrence(cleaned_texts, full_context)

            if not validated_list:
                break

            response_obj = {
                "supporting texts": validated_list
            }
            language_code, language_name = language_detaction(str(validated_list))
            
            response_obj = {
                'data': response_obj,
                'lanCode': language_code
            }
            
            send_quote_response(response_obj)


        return validated_list

    return True







async def extract_clean_points(text):
    regex = r"(?:^|\n)[*-]\s*(.*?)(?:\n|$)"
    
    matches = re.findall(regex, text, re.MULTILINE)
    
    cleaned_points = []
    for match in matches:
        if not match.strip():
            continue
            
        point = re.sub(r"[,.]$", "", match.strip().strip('"\''))
        
        header_pattern = r"^\*\*(.*?)\*\*\s*:"
        header_match = re.search(header_pattern, point)
        if header_match:
            header = header_match.group(1)
            content = point[point.find(':') + 1:].strip()
            if content:
                point = f"{content} ({header})"
            else:
                continue
        
        if point.strip():
            cleaned_points.append(point)
    
    return cleaned_points


# Sort the combined results based on occurrence in original context
def sort_strings_by_text_occurrence(string_list, text_content):
    # Remove commas from text content
    text_content = text_content.replace(",", "")
    search_text = text_content.lower()
    
    # Create a dictionary to store string and its position in text
    positions = {}
    
    for s in string_list:
        # Remove commas from the search string
        s_no_comma = s.replace(",", "")
        search_string = s_no_comma.lower()
        
        # Check if string exists in text
        if search_string in search_text:
            pos = search_text.index(search_string)
            positions[s_no_comma] = pos
    
    # Sort strings based on their position in text
    return sorted(positions.keys(), key=lambda x: positions[x])


def send_quote_response(response):
    try:
        callback_response = request.post(
            url="http://localhost:8000/quote_response",
            json=response,
            headers={"Content-Type": "application/json"}
        )
        if callback_response.status_code == 200:
            print("Quote response sent successfully.")
        else:
            print(f"Failed to send quote response: {callback_response.status_code} - {callback_response.text}")
    except Exception as e:
        print(f"Error sending quote response: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to send quote response")
