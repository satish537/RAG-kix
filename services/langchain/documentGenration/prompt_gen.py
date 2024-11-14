import re, html
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from fastapi.responses import JSONResponse
from fastapi import status
from time import sleep
from utilservice import *

from services.langchain.singleton import OllamaSingleton

ollama_model = OllamaSingleton.get_instance()


PROMPT_TEMPLATE = """
    {documentContent}

    {prompt}

"""


def prompt_template(llm_model: str, input_string: str, prompt: str, return_type: int = 0):
    try:
        prompt = """
        As above given array is topics of document \n    
        Give me prompts for above topics for getting details regarding it from given transcript using ai model \n  
        In Every Prompt Append Text like "From Given Transcript or Context" so It can Generate More Accurate Response. \n
        Response must be in given format like : [["topic name", prompt], ["topic name", prompt], ...]\n   
        Every topics prompt and topic name in one array/list.\n
        Don't change topic name return same as it is given in array \n    
        Generate only one prompt for each title
        """

        prompt_individual = """
        Give me prompt for the above topic related word.\n
        Ensure prompt should be professional.\n
        Generate only one prompt for the topic.\n
        The generated prompt must be in square brackets.
        """
        
        titles_dict, titles_list = separate_titles(input_string)
        response_text = prompt_generator(llm_model, titles_list, prompt)
        titles_dict = extract_prompt(response_text, titles_dict)
        titles_list = missing_prompt(titles_dict)
    
        for i in range(2):
            if len(titles_list) > 5:
                response_text = prompt_generator(llm_model, titles_list, prompt)
                titles_dict = extract_prompt(response_text, titles_dict)
                titles_list = missing_prompt(titles_dict)
    
        for i in range(2):
            if titles_list != []:
                for title in titles_list:
                    response_text = prompt_generator(llm_model, list(title), prompt_individual)
                    titles_dict = individual_prompt(response_text, title, titles_dict)
                    titles_list = missing_prompt(titles_dict)

        if return_type == 0 :
            return JSONResponse(content=titles_dict, status_code=status.HTTP_200_OK)
        else :
            return titles_dict

    except Exception as error:
        handel_exception(error)


def extract_title(title: str):
    title = title.replace("\"", "")

    return title


def remove_special_char(title: str):
    title = title.replace(";", "")
    title = title.replace(":", "")
    title = title.replace("\n", "")
    title = title.replace(",", "")
    title = html.unescape(title)
    title = re.sub(r'\s+', ' ', title).strip()

    return title


def separate_titles(input_string: str):

    titles = re.split(r'\n\*', input_string)
    titles_list = []

    titles_dict = []
    for title in titles:
        if title:
            title_name = title.strip('*').split('\n\t*')[0]
            titles_list.append(remove_special_char(title_name))
            sub_title = [title.strip('*') for title in title.strip('*').split('\n\t*')[1:] if title]
            for title in sub_title:
                titles_list.append(remove_special_char(title))
            titles_dict.append({
                "title": title_name,
                "modify_title": remove_special_char(title_name),
                "prompt": "",
                "children": [
                    {"title": title, "modify_title": remove_special_char(title), "prompt": ""}
                    for title in sub_title
                ]
            })

    return titles_dict, titles_list


def prompt_generator(llm_model: str, titles_list: list, prompt: str):

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt_template = prompt_template.format(documentContent=titles_list, prompt=prompt)

    # llm_model = verify_llm(llm_model)
    # llm = Ollama(model=ollama_model, num_predict=5000)

    response_text = ollama_model.invoke(prompt_template)
    return response_text


def missing_prompt(titles_dict: list):
    missing_titles = []

    for title_obj in titles_dict:
        if title_obj["prompt"] == "" :
            title = title_obj["modify_title"]
            missing_titles.append(title)

        for child in title_obj["children"]:
            if child["prompt"] == "":
                sub_title = child["modify_title"]
                missing_titles.append(sub_title)

    return missing_titles


def extract_prompt(response_text: str, titles_dict: list):

    matches = re.findall(r'\[(.*?)\]', response_text)

    response_list = []
    for m in matches:
    	m = m.replace("'", "")
    	ele = m.split(",", 1)
    	response_list.append(ele)

    for title_obj in titles_dict:
        children = title_obj["children"]
        for response in response_list:
            if len(response) >= 2:
                title = remove_special_char(title_obj["modify_title"])
                prompt_title = extract_title(response[0])
                # if title_obj["prompt"] == "" and find_word_in_sentence(title, response[0]):
                if (title_obj["prompt"] == "" and title == prompt_title) or (title_obj["prompt"] == "" and matching_percentage(title, prompt_title) > 80):
                    title_obj["prompt"] = response[1]
                    response_list.remove(response)
                    break
            
        for i, child in enumerate(children, start=0):
            for response in response_list:
                if len(response) >= 2:
                    prompt_title = extract_title(response[0])
                    sub_title = remove_special_char(child["modify_title"])
                    # if child["prompt"] == "" and find_word_in_sentence(sub_title, response[0]):
                    if (child["prompt"] == "" and sub_title == prompt_title) or (child["prompt"] == "" and matching_percentage(sub_title, prompt_title) > 80):
                        child["prompt"] = response[1]
                        response_list.remove(response)
                        break

    return titles_dict


def individual_prompt(response_text: str, title: str, titles_dict: list):
    matches = re.findall(r'\[(.*?)\]', response_text)
    if len(matches) > 0 :
        for title_obj in titles_dict:
            children = title_obj["children"]
            if title_obj["modify_title"] == title and title_obj["prompt"] == "":
                title_obj["prompt"] = matches[0]
                break
        
            for i, child in enumerate(children, start=0):
                if child["modify_title"] == title and child["prompt"] == "":
                    child["prompt"] = matches[0]
                    break

    return titles_dict





