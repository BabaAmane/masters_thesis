from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm

def calculate_similarity(text1, text2_list, model):
    # テキスト1のベクトルを計算
    vec1 = model.infer_vector(simple_preprocess(text1))

    # テキスト2のベクトルを計算
    vec2_list = [model.infer_vector(simple_preprocess(text2)) for text2 in text2_list]

    # コサイン類似度を計算
    similarities = cosine_similarity([vec1], vec2_list)

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

    # worry_text_list = worry_text_list[:10]
    # famous_quote_list = famous_quote_list[:1000]

    # TaggedDocumentの作成
    tagged_data = [TaggedDocument(words=simple_preprocess(text), tags=[str(i)]) for i, text in tqdm(enumerate(famous_quote_list))]

    # Doc2Vecモデルの学習
    model = Doc2Vec(tagged_data, vector_size=100, min_count=1, epochs=10)

    # リスト1の各テキストに対して最も類似度が高い文章をリスト2から抽出
    most_similar_texts = [calculate_similarity(text1, famous_quote_list, model) for text1 in tqdm(worry_text_list)]

    # print(most_similar_texts)

    # データフレーム化
    df_link = pd.DataFrame(list(zip(worry_text_list, most_similar_texts)), columns = ['worry_text', 'famous_quote'])
    result_dir = args.result_dir
    df_link.to_csv(result_dir, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worry_text_data_dir', type=str, default='../worry_text_data/analysis/by_length/edit_after/process_after/negative_transformer_more9_concat_freq.csv',
                        help='悩みの文章')
    parser.add_argument('--famous_quote_data_dir', type=str, default='../famous_quote_data/final_famous_quote_data/famous_quote_data_condition2_clean.csv',
                        help='名言')
    parser.add_argument('--result_dir', type=str, default='result/linked_negative_text_famous_quote_doc2_vec.csv',
                        help='紐付け結果')

    args = parser.parse_args()
    main(args) 