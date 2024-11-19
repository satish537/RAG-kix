import re

async def extract_single_input(response):
    # Clean the response to remove any trailing backslashes and extra spaces
    response = response.replace("\\", " ")

    # Extract Theme (between "Theme:" and "Description:")
    theme_match = re.search(r"Theme:\s*(.+?)\s*Description:", response, re.DOTALL)
    theme = theme_match.group(1).strip() if theme_match else None

    # Extract Description
    description_match = re.search(r"Description:\s*(.+?)\s*Supporting texts:", response, re.DOTALL)
    description = description_match.group(1).strip() if description_match else None

    # Extract Supporting Texts
    supporting_texts_match = re.search(r"Supporting texts:\s*((?:- .+\n?)+)", response)
    if supporting_texts_match:
        supporting_texts = [
            text.strip('- ').strip('"') for text in supporting_texts_match.group(1).split('\n') if text.strip()
        ]
        supporting_texts = [re.sub(r"[,.]", "", point) for point in supporting_texts]    
    else:
        supporting_texts = []

    # Combine into dictionary
    return {
        "theme": theme,
        "description": description,
        "supporting texts": supporting_texts
    }

async def extract_points_from_regenerated_res(input_string):
    pattern = r'(?:-\s*|\d+\.\s*)"(.*?)"'
    matches = re.findall(pattern, input_string)

    # Clean the points
    cleaned_points = []
    for point in matches:
        point = re.sub(r"[,.0-9]", "", point)
        point = re.sub(r"\s*\([^)]*\)$", "", point)
        cleaned_points.append(point.strip())

    return cleaned_points

async def get_matching_strings(paragraph, string_list):
    matching_strings = []
    for string in string_list:
        # Create a regex pattern to match the whole word
        pattern = r'\b' + re.escape(string) + r'\b'
        # Check if the word is in the paragraph
        if re.search(pattern, paragraph, re.IGNORECASE):  # Case-insensitive match
            matching_strings.append(string)

    return matching_strings


async def extract_points_from_text(response: str) -> list:
    """
    Extracts points from a response, separating them by line breaks or sentences.

    Args:
        response (str): The input text containing points.

    Returns:
        list: A list of extracted points.
    """
    # Split the response into lines based on '\n'
    lines = response.split('\n')
    
    points = []
    for line in lines:
        # Remove leading numbers followed by a dot (e.g., '1.', '2.') and trim whitespace
        clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
        
        # Filter out empty lines and add valid points to the list
        if clean_line:
            points.append(clean_line)
    
    # Further split any overly long point into individual sentences if necessary
    final_points = []
    for point in points:
        # Split on '. ' to separate sentences
        sentences = point.split('. ')
        final_points.extend([sentence.strip() for sentence in sentences if sentence.strip()])
    
    return final_points








# Back-up

# async def extract_single_input(response):
#     # Clean the response to remove any trailing backslashes and extra spaces
#     response = response.replace("\\", " ")

#     # Extract Theme (between "Theme:" and "Description:")
#     theme_match = re.search(r"Theme:\s*(.+?)\s*Description:", response, re.DOTALL)
#     theme = theme_match.group(1).strip() if theme_match else None

#     # Extract Description
#     description_match = re.search(r"Description:\s*(.+?)\s*Supporting texts:", response, re.DOTALL)
#     description = description_match.group(1).strip() if description_match else None

#     # Extract Supporting Texts
#     supporting_texts_match = re.search(r"Supporting texts:\s*((?:- .+\n?)+)", response)
#     if supporting_texts_match:
#         supporting_texts = [
#             text.strip('- ').strip('"') for text in supporting_texts_match.group(1).split('\n') if text.strip()
#         ]
#     else:
#         supporting_texts = []

#     # Combine into dictionary
#     return {
#         "theme": theme,
#         "description": description,
#         "supporting texts": supporting_texts
#     }