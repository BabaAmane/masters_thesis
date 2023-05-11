import gensim
import numpy as np
import MeCab
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
import gc

def encode_sentences(model, sentences):
        encoded = model.encode(sentences)
        return encoded / np.linalg.norm(encoded, axis=1, keepdims=True)

def calculate_most_similar_sentences(encoded_worry_text_list: list, encoded_famous_quote_list: list, famous_quote_list: list)-> list:
    # コサイン類似度を計算する
    similarity_matrix = np.dot(encoded_worry_text_list, encoded_famous_quote_list.T)

    # 最も類似度が高い文章を出力する
    most_similar_indices = np.argmax(similarity_matrix, axis=1)
    most_similar_sentences = [famous_quote_list[i] for i in most_similar_indices]
    return most_similar_sentences
   

def main():    
    # 悩みのデータ読み込み
    worry_text_data_dir = '../worry_text_data/user_text_data_negative_judgment/user_text_data_negative_judgment.csv'
    df_worry_text_data = pd.read_csv(worry_text_data_dir)
    df_worry_text_data = df_worry_text_data.dropna(how='any')

    # 名言のデータの読み込み
    famous_quote_data_dir = '../famous_quote_data/final_famous_quote_data/famous_quote_data_append_conditions2.csv'
    df_famous_quote_data = pd.read_csv(famous_quote_data_dir)
    df_famous_quote_data = df_famous_quote_data.dropna(how='any')

    worry_text_list = df_worry_text_data['message'].to_list()
    famous_quote_list = df_famous_quote_data['famous_quote'].to_list()

    print('worry text length', len(worry_text_list))
    print('famous text length', len(famous_quote_list))

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # データを分割
    worry_text_list1 = worry_text_list[:120000]
   
    # worry_text_listとfamous_quote_listの文章をエンコードする
    start_time = time.time()
    encoded_worry_text_list1 = encode_sentences(model, worry_text_list1)
    end_time = time.time()
    # 計算にかかった時間を出力
    elapsed_time = end_time - start_time
    print("encode worry text Elapsed time:", elapsed_time, "seconds")
    
    start_time = time.time()
    encoded_famous_quote_list = encode_sentences(model,famous_quote_list)

    end_time = time.time()
    # 計算にかかった時間を出力
    elapsed_time = end_time - start_time
    print("encode famous quote Elapsed time:", elapsed_time, "seconds")
    print('encode終了')

    # 計算1個目
    most_similar_sentences1 = calculate_most_similar_sentences(encoded_worry_text_list1, encoded_famous_quote_list, famous_quote_list)

    gc.collect()
    del worry_text_list1
    gc.collect()

    worry_text_list2 = worry_text_list[120000:]
    encoded_worry_text_list2 = encode_sentences(model, worry_text_list2)

    # 計算1個目
    most_similar_sentences2 = calculate_most_similar_sentences(encoded_worry_text_list2, encoded_famous_quote_list, famous_quote_list)

    most_similar_sentences = most_similar_sentences1 + most_similar_sentences2

    
    # データフレーム化
    df_link = pd.DataFrame(list(zip(worry_text_list, most_similar_sentences)), columns = ['worry_text', 'famous_quote'])
    df_link.to_csv('result/linked_negative_text_famous_quote.csv', index=False)

if __name__=='__main__':
    main()