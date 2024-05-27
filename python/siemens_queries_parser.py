import json 

input_file = "C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/eval.txt"
output_file = "Siemens/queries.jsonl"

def process_queries(input_path, output_path): 
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Splitting the content based on the specified delimiter (three backspaces)
    queries = content.split('\n\n\n')  

    # Open the output file to write
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, query in enumerate(queries, start=1):
            # Trim whitespace and construct the JSON object
            query_data = {
                "_id": f"genQ{idx}",
                "text": query.strip(),
                "metadata": {}
            }
            # Write the JSON object to the file as a single line
            json.dump(query_data, outfile, ensure_ascii=False)
            outfile.write('\n')  # New line for the next JSON object

# Call the function with the appropriate file paths
process_queries(input_file, output_file)