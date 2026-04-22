import os
from tokenizer.tokenizer_utils import get_tokenizer


def merge_txt_from_folder(folder_path):
    merged_text = ""

    # txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    txt_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".txt") and f != "dataset.txt"
    ]
    txt_files.sort()

    for idx, file_name in enumerate(txt_files, start=1):
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        merged_text += f"Title txt file {idx}: {file_name}\n"
        merged_text += f"content:\n{content}\n"
        merged_text += "\n" + "="*50 + "\n\n"

    return merged_text


def get_text_data(save=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    merged_content = merge_txt_from_folder(current_dir)

    if save:
        output_path = os.path.join(current_dir, "dataset.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(merged_content)

    return merged_content


def analyze_text(text):
    tokenizer = get_tokenizer()
    total_characters = len(text)
    total_tokens = len(tokenizer.encode(text))

    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    return total_tokens , total_characters