#from regrex_book.json print the strings
import json

with open('regrex_book.json', 'r') as f:
    data = json.load(f)
    print(data['extracted_text'])