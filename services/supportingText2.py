import asyncio, math
import os, time, re, aiohttp
import numpy as np
from utilservice import *
from torch import threshold
from typing import List, Dict
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

threshold_value = 0.7


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

async def process_result_group(result_group, context, theme, description):
    # print(result_group)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in result_group])

    cleaned_lines = []
    for line in context_text.split('\n'):
        if line.startswith("Current Date and Time") or line.startswith("Current User's Login"):
            line = line.replace(",", "")
        cleaned_lines.append(line)
    
    cleaned_context = '\n'.join(cleaned_lines)
    supporting_text_list = await generate_supporting_text(cleaned_context, theme, description)

    cleaned_texts = []
    for text in supporting_text_list:
        if not text.strip():
            continue
            
        cleaned_text = re.sub(r"[,.]$", "", text.strip().strip('"\''))
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

    validated_list = await sort_strings_by_text_occurrence(cleaned_texts, context)
    
    return validated_list

async def sort_strings_by_text_occurrence(string_list, text_content):
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


async def generate_quotes_details(projectID: str, questionId, participantId, prompt, theme, description, metadata, startKValue, endKValue, percentage: float = 50):
    print(f"Generating quotes details for projectID: {projectID} startKValue: {startKValue}, endKValue: {endKValue}")
    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    filter_obj = await build_chroma_filter(
        projectID=projectID,
        questionId=questionId,
        participantId=participantId,
        metadata=metadata
    )

    all_results = db.similarity_search_with_score(prompt, k=endKValue, filter=filter_obj)
    results = all_results[startKValue:endKValue]

    try:
        kValueReached = True
        filtered_chunks = db._collection.get(where=filter_obj, include=["documents"])
        filtered_count = len(filtered_chunks["documents"])
        if filtered_count > endKValue + 2:
            kValueReached = False
    except Exception as e:
        print(f"Error retrieving filtered chunks: {e}")
        filtered_count = 0
        kValueReached = False
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"projectId '{projectID}' does not have any matching data"
        )

    group_size = 2
    result_groups = [results[i:i + group_size] for i in range(0, len(results), group_size)]
    all_supporting_texts = []
    full_context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    with ThreadPoolExecutor(max_workers=math.ceil(len(results)/2)) as executor:
        futures = [
            asyncio.create_task(process_result_group(group, full_context, theme, description))
            for group in result_groups
        ]
        completed_results = await asyncio.gather(*futures)
        for result in completed_results:
            if result:
                all_supporting_texts.extend(result)


    response_obj = {
        "supporting texts": all_supporting_texts,
        
    }
    language_code, language_name = language_detaction(str(response_obj))

    response_obj = {
        'data': response_obj,
        'kValueReached': kValueReached,
        'lanCode': language_code
    }

    print("\n")
    print(response_obj)

    return response_obj





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




