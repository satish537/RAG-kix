from llama_index.embeddings.ollama import OllamaEmbedding


def get_embedding_function():
    
    ollama_embedding = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )

    return ollama_embedding



# pass_embedding = ollama_embedding.get_text_embedding_batch(
#     ["This is a passage!", "This is another passage"], show_progress=True
# )
# print(pass_embedding)

# query_embedding = ollama_embedding.get_query_embedding("Where is blue?")
# print(query_embedding)