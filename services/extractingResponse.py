import re

# Example single input
response = """
Theme: Localized Branding and Strategy in Foreign Markets 
Description: Discussing the adaptive approach to branding and strategy for clients in foreign markets, emphasizing cultural sensitivity and local market understanding.
Supporting texts:
- "We have to understand that every market is unique, and you can't just take what works in one market and apply it to another."
- "You need to be culturally sensitive when working with a different market; the way they perceive brands and do business might be vastly different from your home market."
- "For example, if we're working in Asia, we have to consider factors like collectivism and hierarchical societies. This can greatly influence how a brand is perceived and received in that region."
"""

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


# Extract and print the result
extracted_data = extract_single_input(response)
print(extracted_data)






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