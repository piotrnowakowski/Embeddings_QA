import json
import re

def preprocess_body(body):
    body_plain = re.sub('<.*?>', '', body)
    return body_plain

def extract_key_value_pairs(text):
    pattern = r'(\b[A-Za-z\s]+:)\s*([^<\n]+)'
    matches = re.findall(pattern, text)
    extracted_data = {key[:-1]: value.strip() for key, value in matches}
    return extracted_data

def process_data(data):
    structured_data = []
    for item in data:
        body_plain = preprocess_body(item['body'])
        combined_text = item['title'] + ". " + body_plain
        key_value_pairs = extract_key_value_pairs(combined_text)
        structured_data.append(key_value_pairs)
    return structured_data

# Read JSON data from the 'documentation.json' file
with open('documentation/documentation.json', 'r') as file:
    data = json.load(file)
print('done')
# Process the data
structured_data = process_data(data)

# Save the processed data to the 'processed_embeddings.json' file
with open('processed_embeddings_splitted.json', 'w') as file:
    json.dump(structured_data, file, indent=2)
