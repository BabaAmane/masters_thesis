import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

def main():
    df_concat = pd.read_csv('final_famous_quote_data/famous_quote_data_append_conditions2.csv')

    df_data1 = df_concat[:57708]
    df_data2 = df_concat[57708:]

    train_texts = list(df_data1['famous_quote'])
    test_texts = list(df_data2['famous_quote'])

    # train_texts = train_texts[:10]
    # test_texts = test_texts[:5]

    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    features = []
    for text in tqdm(train_texts):
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            outputs = model(input_ids)
            pooled_output = outputs[1].numpy().tolist()
            features.append(pooled_output)

    test_features = []
    for text in tqdm(test_texts):
        input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            outputs = model(input_ids)
            pooled_output = outputs[1].numpy().tolist()
            test_features.append(pooled_output)

    features = np.array(features)
    features = features.reshape(len(train_texts), -1)
    test_features = np.array(test_features)

    max_similarity_list = []
    hit_quotes_list = []
    for i, test_feature in tqdm(enumerate(test_features)):
        test_feature = test_feature.reshape(1, -1)
        similarities = cosine_similarity(features, test_feature)
        max_similarity = max(similarities)
        max_index = similarities.argmax()
        # print(f"テキスト: {test_texts[i]}")
        # print(f"類似度: {max_similarity}")
        # print(f"評価: {train_texts[max_index]}")
        max_similarity_list.append(max_similarity[0])
        hit_quotes_list.append(train_texts[max_index])

    df_result = pd.DataFrame(list(zip(test_texts, hit_quotes_list, max_similarity_list)), columns = ['famous_quote', 'hit_famous_quote', 'max_similarity'])
    df_result.to_csv('data2_quality_same_data1/data2_quality_same_data1.csv', index=False)


if __name__=='__main__':
    main()
