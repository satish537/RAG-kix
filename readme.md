install Anaconda and setup a conda environment

https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment

install required packages:
* langchain
* fastapi
* aiofiles
* llamaindex


# install ollama

https://ollama.com/download

pull mistral and run with following command in the terminal

create 2 empty directories at the root level
```
data
chroma
```

create 2 empty directories in the chroma directory
```
confluence
langchain
llamaindex
```


run the application with following command in the terminal
```
uvicorn api:app --reload --port=8000
```


visit the below link for access fast api docs
```
http://localhost:8000/docs
```
