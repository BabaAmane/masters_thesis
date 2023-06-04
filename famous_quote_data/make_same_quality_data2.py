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

    # train_texts = train_texts[:100]
    # test_texts = test_texts[:5]

    model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
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
    train_features_mean = np.mean(features, axis=0).reshape(1, -1)
    test_features = np.array(test_features)

  
    similarities_list = []
    for i, test_feature in tqdm(enumerate(test_features)):
        test_feature = test_feature.reshape(1, -1)
        similarities = cosine_similarity(test_feature, train_features_mean)

        similarities_list.append(similarities)

    similarities_array = np.concatenate(similarities_list, axis=0)

    similarities_list = []

    for i, similarities in enumerate(similarities_array):
        similarities_list.append(similarities[0])

    # データフレーム保存
    df_result = pd.DataFrame(list(zip(test_texts, similarities_list)), columns = ['famous_quote', 'similarity'])
    df_result.to_csv('data2_quality_same_data1/data2_quality_same_data1_mean.csv', index=False)
                


        

if __name__=='__main__':
    main()
