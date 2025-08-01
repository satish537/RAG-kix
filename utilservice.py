import os, re, shutil
from datetime import datetime
from langdetect import detect
import aiofiles

DIRECTORY_PATH = "data"

language_code = {
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'gl': 'Galician',
    'de': 'German',
    'el': 'Greek',
    'gu': 'Gujarati',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'id': 'Indonesian',
    'ga': 'Irish',
    'it': 'Italian',
    'ja': 'Japanese',
    'kn': 'Kannada',
    'kk': 'Kazakh',
    'ko': 'Korean',
    'ku': 'Kurdish',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
    'mk': 'Macedonian',
    'ms': 'Malay',
    'mt': 'Maltese',
    'ne': 'Nepali',
    'no': 'Norwegian',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sr': 'Serbian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'es': 'Spanish',
    'sw': 'Swahili',
    'sv': 'Swedish',
    'ta': 'Tamil',
    'te': 'Telugu',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'cy': 'Welsh',
    'xh': 'Xhosa',
    'yi': 'Yiddish',
    'zu': 'Zulu'
}




# change document name
async def rename_and_save_file(file_obj, document_name: str = None, version_id: str = None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_extension = os.path.splitext(file_obj.filename)[1]

    if document_name and version_id:
        new_file_name = f"{current_time}_{document_name}_v{version_id}{file_extension}"
    else:
        new_file_name = f"{current_time}{file_extension}"

    #new_file_name = os.path.basename(new_file_name)
    new_file_path = os.path.join(DIRECTORY_PATH, new_file_name)
    if not os.path.exists(DIRECTORY_PATH):
        os.makedirs(DIRECTORY_PATH)

    async with aiofiles.open(new_file_path, "wb") as f:
        content = await file_obj.read()
        await f.write(content)
    return new_file_path, new_file_name


# Delete document/file
def delete_document(path: str, filename: str = None):
    try:
        file_path = os.path.join(path, filename) if filename else path
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"File {file_path} has been deleted successfully.")
            return True
        else:
            print(f"File {file_path} does not exist.")
            return False

    except Exception as e:
        print(f"An error occurred while trying to delete the file: {e}")
        return False


# Delete Directory
def delete_directory(directory_path):
    try:
        # Check if the directory exists
        if os.path.exists(directory_path):
            # Delete the directory and its contents
            shutil.rmtree(directory_path)
            # print(f"Directory '{directory_path}' has been deleted successfully.")
            return True
        else:
            print(f"Directory '{directory_path}' does not exist.")
            return False
    except Exception as e:
        print(f"An error occurred while deleting the directory: {e}")
        return False


def language_detaction(text: str):
    try:
        res_langcode = detect(text)
        res_language = language_code.get(res_langcode, "Unknown Language")
    except Exception as e:
        res_langcode = "und"
        res_language = "Can't Detected"
        print(e)

    return res_langcode, res_language


