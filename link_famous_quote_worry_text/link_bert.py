import torch
from transformers import BertJapaneseTokenizer, BertModel
import pandas as pd
import argparse
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def calculate_similarity(text1, text2_list):
    # BERTモデルとトークナイザーの読み込み
    model_name = 'bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # テキスト1の特徴量を計算
    input_ids1 = tokenizer.encode(text1, add_special_tokens=True, truncation=True, padding='max_length', max_length=512)
    input_ids1 = torch.tensor([input_ids1])
    with torch.no_grad():
        outputs1 = model(input_ids1)
        pooled_output1 = outputs1[1].numpy()

    # テキスト2の特徴量を計算
    input_ids2_list = [tokenizer.encode(text2, add_special_tokens=True, truncation=True, padding='max_length', max_length=512) for text2 in text2_list]
    input_ids2_list = torch.tensor(input_ids2_list)
    with torch.no_grad():
        outputs2 = model(input_ids2_list)
        pooled_output2 = outputs2[1].numpy()

    # コサイン類似度を計算
    similarities = cosine_similarity(pooled_output1.reshape(1, -1), pooled_output2)

    # 最も類似度が高い文章を抽出
    max_index = np.argmax(similarities)
    most_similar_text = text2_list[max_index]

    return most_similar_text


def main(args):
    # 悩みのデータ読み込み
    worry_text_data_dir = args.worry_text_data_dir
    df_worry_text_data = pd.read_csv(worry_text_data_dir)
    df_worry_text_data = df_worry_text_data.dropna(how='any')

    # 名言のデータの読み込み
    famous_quote_data_dir = args.famous_quote_data_dir
    df_famous_quote_data = pd.read_csv(famous_quote_data_dir)
    df_famous_quote_data = df_famous_quote_data.dropna(how='any')

    worry_text_list = df_worry_text_data['message'].to_list()
    famous_quote_list = df_famous_quote_data['famous_quote'].to_list()

    print('worry text length', len(worry_text_list))
    print('famous text length', len(famous_quote_list))

    worry_text_list = worry_text_list[:5]


    # リスト1の各テキストに対して最も類似度が高い文章をリスト2から抽出
    most_similar_texts = [calculate_similarity(worry_text, famous_quote_list) for worry_text in worry_text_list]


    df_link = pd.DataFrame(list(zip(worry_text_list, most_similar_texts)), columns = ['worry_text', 'famous_quote'])
    result_dir = args.result_dir
    df_link.to_csv(result_dir, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worry_text_data_dir', type=str, default='../worry_text_data/analysis/by_length/edit_after/process_after/negative_transformer_more9_concat.csv',
                        help='悩みの文章')
    parser.add_argument('--famous_quote_data_dir', type=str, default='../famous_quote_data/final_famous_quote_data/famous_quote_data_append_conditions2.csv',
                        help='名言')
    parser.add_argument('--result_dir', type=str, default='result/linked_negative_text_famous_quote.csv',
                        help='紐付け結果')

    args = parser.parse_args()
    main(args) 

    