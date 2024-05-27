import json

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def generate_json(text):
    #Splitted text by 3 newlines
    paragraphs = text.split('\n\n\n')  
    json_objects = []
    for idx, paragraph in enumerate(paragraphs):
        # Create JSON object with text, title, and _id
        json_obj = {
            "text": paragraph.strip(),  # Remove leading/trailing whitespaces
            "title": "",
            "_id": str(idx + 1)  # Chronological order starting from 1
        }
        json_objects.append(json_obj)
    return json_objects

def write_jsonl_file(json_objects, output_file):
    with open(output_file, 'w') as file:
        for obj in json_objects:
            file.write(json.dumps(obj, ensure_ascii=False))
            file.write('\n')

def main(input_file, output_file):
    text = read_text_file(input_file)
    json_objects = generate_json(text)
    write_jsonl_file(json_objects, output_file)

if __name__ == "__main__":
    data_path = "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/handpicked.txt"
    input_file = data_path   
    output_file = "Siemens/corpus.jsonl"   
    main(input_file, output_file)