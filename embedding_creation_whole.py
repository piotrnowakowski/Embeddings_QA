import json
from bs4 import BeautifulSoup
import re

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, 'html.parser')
    cleaned_text = soup.get_text(separator=' ')  # replace HTML tags with spaces
    cleaned_text = cleaned_text.replace('\n', ' ')  # replace newlines with spaces
    cleaned_text = cleaned_text.replace(':', ': ')  # ensure space after colon
    cleaned_text = ' '.join(cleaned_text.split())  # remove extra spaces
    return cleaned_text

def create_embeddings():
    # Read data from file
    with open('documentation/documentation.json', 'r') as f:
        data_json = json.load(f)

    processed_data = []

    for item in data_json:
        cleaned_body = clean_html(item['body'])
        combined_text = item['title'] + ' ' + cleaned_body
        processed_data.append({
            'title': item['title'],
            'combined_text': combined_text
        })

    def extract_info(article, state):
        state = state.split(':')[0]
        text = article['combined_text']
        mcc = re.search('MCC\s?:\s?(\d+)', text)
        dial_code = re.search('Dial Code\s?:\s?(\d+)', text)
        alphanumeric = re.search('((?:Alphanumeric|Alpha numeric|alpha numeric).*?)(?:\.|$)', text, re.IGNORECASE)
        
        return {
            'title': state,
            'MCC': mcc.group(1) if mcc else None,
            'Dial Code': dial_code.group(1) if dial_code else None,
            'Alphanumeric': alphanumeric.group(1) if alphanumeric else None,
            'combined_text': text
        }


    parsed_data = [extract_info(article, article['title']) for article in processed_data]
    #parsed_data = {article['title']: extract_info(article) for article in data}

    # Save to a JSON file
    with open('documentation\parsed_data.json', 'w') as f:
        json.dump(parsed_data, f, indent=2)

create_embeddings()