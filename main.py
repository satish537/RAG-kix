import uvicorn
import os

# Set the environment variable in Python

os.environ["OLLAMA_NUM_THREADS"] = "4"
# os.environ["OLLANA_NUM_PARALLEL"]="4"
# Verify that the environment variable is set
print("OLLAMA_NUM_THREADS is set to:", os.getenv("OLLAMA_NUM_THREADS"))

uvicorn.run("api:app", host="0.0.0.0", port=3001)

# chmod o+w .
# command for give permission to the current directory of accessing the files