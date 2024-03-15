from openai import OpenAI
import pandas as pd
from tqdm import tqdm

API_KEY = 'LL-vzFVzONMBqahGzEJO3hCAPhy2RM1KyVx17BidVwcLA8qgtWACmhD9W99GePVoAaE'

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.llama-api.com"
)

def openapi_test(topic):
    response = client.chat.completions.create(
        model='mistral-7b-instruct',
        messages=[
            {"role": "system", "content": "Write 600-800 words for a given topic"},
            {"role": "user", "content": "Topic: " + topic}
        ],
        seed=78,
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    for file_name in ['data_mining', 'geometry', 'physics', 'precalculus']:
        with open(f'AL-CPL-dataset-master/data/{file_name}.pairs', encoding='utf-8') as f:
            lines = [l[:-1].split(',') for l in f.readlines()]

            for l in lines:
                if len(l) != 2:
                    print(f'{file_name} - Pair Error - {l}')

            temp = list(set([topic for pair in lines for topic in pair]))
            descriptions = {}
            error_count = 0

            for topic in tqdm(temp):
                try:
                    content = openapi_test(topic)
                except:
                    content = 'error'
                    error_count += 1
                descriptions[topic] = [content]

            print(f'{file_name} error count: {error_count}')

            data = pd.DataFrame.from_dict(descriptions, orient='index')
            data.to_csv(f'generated_data/{file_name}.csv')
