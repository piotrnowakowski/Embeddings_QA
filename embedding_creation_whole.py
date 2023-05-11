import json
from bs4 import BeautifulSoup

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    return soup.get_text()

# Read data from file
with open('documentation.json', 'r') as f:
    data_json = json.load(f)

processed_data = []

for item in data_json:
    cleaned_body = clean_html(item['body'])
    combined_text = item['title'] + ' ' + cleaned_body
    processed_data.append({
        'title': item['title'],
        'combined_text': combined_text
    })

# Save processed data to file
with open('processed_embeddings_whole.json', 'w') as f:
    json.dump(processed_data, f, indent=4)
