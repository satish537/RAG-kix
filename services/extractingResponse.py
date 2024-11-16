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
    else:
        supporting_texts = []

    # Combine into dictionary
    return {
        "theme": theme,
        "description": description,
        "supporting texts": supporting_texts
    }



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