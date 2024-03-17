import pandas as pd
from tqdm import tqdm
from mistralai.client import MistralClient

MISTRAL_KEY = '<YOUR KEY HERE>'
client = MistralClient(api_key=MISTRAL_KEY)


def load_topic_texts():
    data_dict = dict()

    for subject in ['data_mining', 'geometry', 'physics', 'precalculus']:
        dm = pd.read_csv(f'generated_data/{subject}.csv')
        dm.columns = ['topic', 'text']

        for topic, text in zip(dm['topic'].values, dm['text'].values):
            data_dict[topic] = text

    return data_dict


def load_preq_map():
    preq_map = dict()

    for subject in ['data_mining', 'geometry', 'physics', 'precalculus']:
        with open(f'AL-CPL-dataset-master/data/{subject}.preqs', encoding='utf-8') as f:
            lines = [line[:-1].split(',') for line in f.readlines()]

            for line in lines:
                if len(line) != 2:
                    print(f'{subject} - Pair Error - {line}')
                else:
                    if line[0] in preq_map.keys():
                        tmp = preq_map[line[0]]
                        tmp.append(line[1])
                        preq_map[line[0]] = tmp
                    else:
                        preq_map[line[0]] = [line[1]]

    return preq_map


def load_test_set():
    pairs = []

    for file_name in ['data_mining', 'geometry', 'physics', 'precalculus']:
        with open(f'AL-CPL-dataset-master/data/{file_name}.pairs', encoding='utf-8') as f:
            lines = [l[:-1].split(',') for l in f.readlines()]

            for l in lines:
                if len(l) != 2:
                    print(f'{file_name} - Pair Error - {l}')

            pairs += lines

    return pairs


def compare_topics(topic_a, topic_b):
    # model = 'mixtral-8x7b-instruct'
    model = 'open-mixtral-8x7b'
    messages=[
        {"role": "system",
         "content": "I will give you two topics in the next two prompts, Topic A and Topic B. You have to guess if Topic A is needed to understand Topic B. Answer only one word: 'True' or 'False'."},
        {"role": "user", "content": "Topic A: " + topic_a},
        {"role": "user", "content": "Topic B: " + topic_b},
    ]
    response = client.chat(model=model, messages=messages, random_seed=78)

    # response = client.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     seed=78,
    # )

    return response.choices[0].message.content


if __name__ == '__main__':
    topics = load_topic_texts()
    preq_map = load_preq_map()
    test_set = load_test_set()

    preds = []
    true_preds = []
    a_s = []
    b_s = []
    err_cnt = 0
    for t in tqdm(test_set):
        try:
            tpc_b, tpc_a = t[0], t[1]
            a_s.append(tpc_a)
            b_s.append(tpc_b)

            text_a = topics[tpc_a]
            text_b = topics[tpc_b]
            if tpc_a in preq_map:
                true_pred = tpc_b in preq_map[tpc_a]
            else:
                true_pred = False

            true_preds.append(true_pred)
            try:
                response = compare_topics(text_a, text_b)
                preds.append(response)
            except:
                preds.append('llm_error')
                err_cnt += 1
        except:
            preds.append('vague_error')
            err_cnt += 1

        if len(preds) % 100 == 0 and len(preds) > 1:
            pd.DataFrame({'topic_a': a_s, 'topic_b': b_s, 'y_true': true_preds, 'y_pred': preds}).to_csv(
                f'generated_data/preds_chunk_{len(preds)}.csv', index=False)

    print('Error count: ', err_cnt)
    pd.DataFrame({'topic_a': a_s, 'topic_b': b_s, 'y_true': true_preds, 'y_pred': preds}).to_csv('generated_data/preds.csv', index=False)
