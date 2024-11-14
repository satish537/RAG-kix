from langchain_community.llms.ollama import Ollama

class OllamaSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = Ollama(model="llama3.2:3b", keep_alive=-1)
        return cls._instance
