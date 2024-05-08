with open('C:/Users/Siemens/Documents/TFE_Nick_Kuijpers/LoRa_GPL/GPL_LoRA/generated/SAMS/qgen-qrels/train.tsv', 'r') as input_file, open('output.tsv', 'w') as output_file:
    for line in input_file:
        line = line.strip()  # Remove leading and trailing whitespace
        if line:  # Check if the line is not empty
            output_file.write(line + '\n')