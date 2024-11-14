from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    # embeddings = embed = OpenAIEmbeddings(
    #             openai_api_key=OPENAI_API_KEY,
    #             model="text-embedding-ada-002"
    #         )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
