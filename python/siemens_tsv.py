output_file_path = 'test.tsv'

def generate_tsv(output_path, max_corpus_id):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Write the header row
        outfile.write("query-id\tcorpus-id\tscore\n")
        # Generate the rows for the file
        for i in range(1, max_corpus_id + 1):
            query_id = f"genQ{i}"
            corpus_id = i 
            score = 1
            outfile.write(f"{query_id}\t{corpus_id}\t{score}\n")

generate_tsv(output_file_path, 170)