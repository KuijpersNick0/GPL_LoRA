import json

def convert_id_to_string(jsonl_file, output_file):
    with open(jsonl_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            data['_id'] = str(data['_id'])  # Convert _id to string format
            json.dump(data, f_out)
            f_out.write('\n')

# Specify the paths to the input and output files
input_jsonl_file = 'C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/generated/SAMS/corpus.jsonl'  
output_jsonl_file = 'corpus2.jsonl'   

# Convert _id to string format and write the modified JSONL file
convert_id_to_string(input_jsonl_file, output_jsonl_file)
