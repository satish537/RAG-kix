import os, json
from langchain_community.document_loaders import ConfluenceLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema.document import Document
from services.langchain.embedding import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from services.langchain.text_to_query import text_to_query
from langchain.text_splitter import MarkdownHeaderTextSplitter
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from utilservice import *



PROMPT_TEMPLATE = """
The context provided is data from confluence space.
Answer the question based on the context below. 
If the question cannot be answered using the information provided answer
with "It is not clear from the provided data".

Context: {context}

Question: {query}

Answer the question.

Be informative, gentle, and formal. 
Answer:"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)


# loading confluence space in to vectorstore

def compute_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks


def get_confluence_data_as_vector_langchain(url, query, username, api_key, space_key):
    print("langchain")
    loader = ConfluenceLoader(
        url=url,
        username=username,
        api_key=api_key,
        cql=f'space = "{space_key}" AND title = "{query}"',
        include_attachments=True,
        # cql=f'space = "{space_key}"',
        space_key=space_key,
        # limit=5,
        # max_pages=4
    )

    documents = loader.load()
    docs = []

    for document in documents:
        title = document.metadata['title']
        content = document.page_content
        source = document.metadata['source']
        doc_id = document.metadata['id']
        url = url + doc_id

        data = {'title': title,
                'source': source,
                'doc_id': doc_id,
                'url': url,
                'Header 1': '',
                'Header 2': ''
                }

        md_header_splits = markdown_splitter.split_text(content)

        for i, split in enumerate(md_header_splits):
            data['sub_id'] = i
            data.update(split.metadata)

            data['content'] = f"{data['title']}\n\tsubsection:{data['Header 1']}:\n\tsub_subsection:{data['Header 2']}:\n" + split.page_content
            new_doc = Document(page_content=data['content'], metadata=document.metadata)
            docs.append(new_doc)

    return Chroma.from_documents(docs, get_embedding_function())


def query_on_confluence_data_langchain(llm_model, index, query_text):
    results = index.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query_text)

    llm_model = verify_llm(llm_model)
    model = Ollama(model=llm_model)

    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"


    PROMPT_TEMPLATE2 = """ 
    From given above context can you tell me how much percentage the response is matching with the query. Here is my Query and Response, \n
    Query: {query}\nResponse: {response}\n
    Provide your response like,\n - Matching: [only percentage]
    """
    prompt2 = PROMPT_TEMPLATE2.format(query=query_text, response=response_text)
    
    match llm_model:
        case "mistral":
            LLM_MODEL_NAME = "llama2"
        case _:
            LLM_MODEL_NAME = "mistral"

    matchingIndex = text_to_query(LLM_MODEL_NAME, context_text, prompt2)
    matchingIndex = matchingIndex.body.decode("utf-8")

    response_text = {"response": response_text, "hallucinatingPercentage": matchingIndex}

    return JSONResponse(content=response_text, status_code=status.HTTP_200_OK)
