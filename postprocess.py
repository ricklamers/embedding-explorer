import re

def filter_sentences(input_filename, output_filename):
    """Filters low-quality lines from sentences.txt and saves the result."""

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            line = line.strip() 

            # Filter 1: Empty lines
            if not line:
                continue

            # Filter 2: Lines with excessive punctuation or special characters
            if re.search(r"^[^a-zA-Z0-9\s]{3,}$", line):
                continue

            # Filter 3: Lines with mostly capital letters
            if sum(1 for char in line if char.isupper()) / len(line) > 0.7:
                continue

            # Filter 4: Very short lines (Adjust threshold as needed)
            if len(line.split()) < 5:  
                continue

            # Filter 5: Lines starting with specific patterns
            if re.match(r"(==.*==)|(\[\[.*\]\])|(\d+\.)", line): 
                continue

            # If the line passes the filters, write it to the output file
            outfile.write(line + "\n")


# Example usage
input_filename = "sentences.txt"
output_filename = "sentences-processed.txt"
filter_sentences(input_filename, output_filename)
