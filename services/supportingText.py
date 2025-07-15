import asyncio
import os, time, re
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

Analyze the provided theme, description, transcript and extract the exact sentences or phrases that directly answer. Your task is to generate a list of supporting texts that strictly adheres to the following requirements:

Output Requirements:
Exact Match: The extracted content must be verbatim from the transcript and directly relevant to the question.
No Modifications: Do not add ellipses (...), parentheses, or any explanatory notes. The output must consist only of the sentences or phrases exactly as they appear in the transcript.
List Format: Provide the supporting texts in a list format, one item per line, without including the question or any additional text.
Clarity and Relevance: Each line must correspond to a unique, relevant point from the transcript, ensuring it answers the question.
No Redundancy: Avoid repeating the same sentence or phrase more than once.

Final Output Format:
[Exact matching text block from the transcript]
[Exact matching text block from the transcript]
[Exact matching text block from the transcript]

Input:

===============================================================================================================================

1. Theme Title: {theme}

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
        # print(response_text)
        supporting_text_list = await extract_clean_points(response_text)
        supporting_text_list = await get_matching_strings(context_text, supporting_text_list)
        aggregated_points.extend(supporting_text_list)

        if len(supporting_text_list) >= 3:
            break

    if len(supporting_text_list) < 3:
        supporting_text_list = aggregated_points

    supporting_text_list = list(set(supporting_text_list))
    supporting_text_list = [element for element in supporting_text_list if element and element.strip()]

    return supporting_text_list 



async def generate_quotes_details(projectID: str, questionId, participantId, theme, description, percentage: float = 50):
    if not os.path.exists(CHROMA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project_id '{projectID}' Not Found")

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Constants
    TOP_K_PER_ID = 3
    PERCENTAGE_THRESHOLD = percentage / 100  # Convert percentage to fraction

    # Step 1: Retrieve all relevant documents
    if questionId and participantId:
        results = db.get(include=["documents", "metadatas"], where={"$and": [{"questionId": questionId}, {"participantId": participantId}]})
    else:
        results = db.get(include=["documents", "metadatas"], where={"projectId": projectID})


    if not results or not results.get("documents"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"projectId '{projectID}' does not have any matching data")

    print("🔹 Total Retrieved Chunks from DB:", len(results["documents"]))

    # Extract documents and metadata
    documents = results["documents"]
    metadatas = results["metadatas"]

    if len(documents) != len(metadatas):
        raise ValueError("Mismatch between documents and metadata!")

    input_text = theme + description
    # Initialize embedding function
    embedding_function = get_embedding_function()
    prompt_embedding = np.array(embedding_function.embed_query(input_text))  # ✅ Fix prompt embedding

    scored_chunks = []
    for doc, meta in zip(documents, metadatas):
        chunk_id = meta.get("id", "default")
        doc_embedding = np.array(embedding_function.embed_query(doc))  # ✅ Fix document embedding
        score = 1 - cosine(prompt_embedding, doc_embedding)  # Compute cosine similarity
        scored_chunks.append((doc, meta, chunk_id, score))

    # Step 3: Extract Top-K per unique 'id'
    grouped_chunks: Dict[str, List] = {}
    for doc, meta, chunk_id, score in scored_chunks:
        grouped_chunks.setdefault(chunk_id, []).append((doc, meta, score))

    print(f"🔹 Total Unique Chunk IDs: {len(grouped_chunks)}")  # ✅ Added print for unique IDs

    for chunk_id, chunks in grouped_chunks.items():
        chunks.sort(key=lambda x: x[2], reverse=True)  # Sort by score (high to low)
        grouped_chunks[chunk_id] = chunks[:TOP_K_PER_ID]  # Keep only top N per id

    # Step 4: Collect all top chunks into a single list
    all_top_chunks = [chunk for chunks in grouped_chunks.values() for chunk in chunks]

    # Step 5: Select top X% of scores
    all_top_chunks.sort(key=lambda x: x[2])  # Sort by score (ascending)
    num_selected = max(1, int(len(all_top_chunks) * PERCENTAGE_THRESHOLD))  # Ensure at least 1 chunk
    filtered_chunks = all_top_chunks[-num_selected:]  # Take top X% (highest scores)

    print(f"🔹 Filtered Chunks After Applying {percentage}% Threshold: {len(filtered_chunks)}")

    if not filtered_chunks:
        print(f"⚠️ No chunks met the percentage threshold for Project ID: {projectID}")
        return []

    # Step 6: Process results in batches (Max size 3)
    batches = [filtered_chunks[i:i+3] for i in range(0, len(filtered_chunks), 3)]

    print(f"🔹 Total Batches After Filtering: {len(batches)}")

    async def process_batch(batch):
        context_text = "\n\n---\n\n".join([doc for doc, _, _ in batch])
        return await generate_supporting_text(context_text, theme, description)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        results = await asyncio.gather(*[loop.run_in_executor(executor, asyncio.run, process_batch(batch)) for batch in batches])

    # Flatten results
    supporting_text_list = [item for sublist in results for item in sublist]

    response_obj = {
        "supporting texts": supporting_text_list
    }

    language_code, language_name = language_detaction(str(response_obj))

    response_obj = {
        'data': response_obj,
        'lanCode': language_code
    }

    return response_obj







async def extract_clean_points(text):
    # Regex to match points starting with *, ., -, or numbers (with or without a period or parenthesis after the number)
    regex = r"(?:(?:[-*.\s]*\d*[.)]?\s*)|(?:[*.\-]\s*))(.*)"
    
    # Extract matches
    matches = re.findall(regex, text)
    
    # Clean matches by removing surrounding quotes, whitespace, commas, and dots
    cleaned_points = [re.sub(r"[,.]", "", match.strip().strip('"\'"')) for match in matches]
    
    return cleaned_points




