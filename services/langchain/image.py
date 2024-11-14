# Import necessary libraries
import os, io, fitz
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from fastapi import HTTPException
from utilservice import *

DATA_PATH = "././data"


# model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5-int4', trust_remote_code=True)

 
# Load the model and processor
# model_simalarity = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor_simalarity = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def imagesToPrompt(llm_model: str, filename: str, recording_id: str, prompt_obj: dict):
    # pdf_path = f"{DATA_PATH}/{filename}"
    # output_folder = f"{DATA_PATH}/{recording_id}"
    # extracted_images_dir = extract_images_from_pdf(pdf_path, output_folder)
    # number_of_images = count_images(extracted_images_dir)

    # response_obj = {}
    # for category, prompt in prompt_obj.items():
    #     if type(prompt) is dict:
    #         images = compare_images_in_directory(extracted_images_dir, prompt['prompt'], processor_simalarity, model_simalarity)
    #         response = process_images_with_multiple_prompts(images, category, prompt['prompt'])
    #         response_obj[category] = response
    #     elif type(prompt) is str:
    #         images = compare_images_in_directory(extracted_images_dir, prompt, processor_simalarity, model_simalarity)
    #         response = process_images_with_multiple_prompts(images, category, prompt)
    #         response_obj[category] = response

    # delete_document(DATA_PATH, filename)
    # return response_obj
    return True




def count_images(directory, extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']):
    image_count = 0
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_count += 1
    return image_count

def extract_images_from_pdf(pdf_path, output_folder):
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {output_folder}") from e
    else:
        raise RuntimeError(f"Directory already exists: {output_folder}")

    # Open the PDF file
    pdf_file = fitz.open(pdf_path)
    
    # Iterate over all the pages
    for page_number in range(len(pdf_file)):
        # Get the page
        page = pdf_file.load_page(page_number)
        images = page.get_images(full=True)

        # Iterate over all images on the page
        for img_index, img_info in enumerate(images):
            # Extract image information
            xref = img_info[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Check if the image format is already PNG
            if image_ext.lower() in ["jpeg", "jpg", "png", "bmp", "gif"]:
                try:
                    # Load the image into a Pillow Image object
                    image = Image.open(io.BytesIO(image_bytes))

                    # Define the PNG file name
                    image_name = f"page_{page_number+1}_img_{img_index+1}.png"
                    image_path = os.path.join(output_folder, image_name)

                    # Save the image in PNG format
                    image.save(image_path, "PNG")

                    print(f"Image saved as PNG: {image_path}")
                except Exception as e:
                    print(f"Failed to process image on page {page_number+1}, image {img_index+1}: {e}")
            else:
                print(f"Unsupported image format: {image_ext} on page {page_number+1}, image {img_index+1}")

    # Close the PDF file
    pdf_file.close()

    return output_folder



def answer_image_question(image_path, question, processor, model):
    """
    This function takes an image path, a question, a text processor, and a model as input
    and returns the similarity score between the image and the question.
    """
    # Load the image and preprocess it
    image = Image.open(image_path)
    # Prepare the question as a list
    inputs = processor(text=[question], images=image, return_tensors="pt")
    
    # Get the image and text features
    image_features = inputs.pixel_values
    text_features = inputs.input_ids
    
    # Get the similarity between the image and text features
    outputs = model(**inputs)
    logits_per_image, logits_per_text = outputs.logits_per_image, outputs.logits_per_text
    
    # Take the dot product between the image and text features
    similarity = logits_per_image[0, 0]
    
    return similarity
 
def compare_images_in_directory(image_dir, question, processor, model, num_most_similar=2):
    """
    This function takes a directory path, a question, a text processor, a model,
    and the number of most similar images (default 2) as input and returns a list of
    image paths with the highest similarity scores.
    """
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Store similarities in a dictionary with image paths as keys
    similarities = {image_path: answer_image_question(image_path, question, processor, model) for image_path in image_paths}
    
    # Sort the dictionary by similarity (descending) and get the first `num_most_similar` elements
    most_similar_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:num_most_similar]
    
    # Extract image paths from the sorted list
    most_similar_image_paths = [image_path for image_path, _ in most_similar_images]
    
    return most_similar_image_paths



def process_images_with_multiple_prompts(image_files, category, question, language_code="en"):
    prompts_response = {}

    images = [Image.open(file).convert('RGB') for file in image_files]
 
    msgs = [{'role': 'user', 'content': images + [question]}]
 
    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    language_code, language_name = language_detaction(answer)
    response = {'data': answer, 'lanCode': language_code}

    return response
