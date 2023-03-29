import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import operator
import typing
import functools


class CosineSimilarity:
    """余弦による文字列の類似度を計算する"""
    Vector = typing.List[float]

    @staticmethod
    def dot_product(vec1: Vector, vec2: Vector, sum=sum, map=map, mul=operator.mul) -> float:
        """ベクトルの内積を計算する"""
        return sum(map(operator.mul, vec1, vec2))

    @staticmethod
    def norm(vec: Vector, sum=sum, map=map, mul=operator.mul) -> float:
        """ベクトルのEuclidノルムを計算する"""
        return math.sqrt(sum(map(operator.mul, vec, vec)))

    def cosine(self, vec1: Vector, vec2: Vector) -> float:
        """ベクトルのなす角の余弦を計算する"""
        return self.dot_product(vec1, vec2) / (self.norm(vec1) * self.norm(vec2))

    def __call__(self, a: str, b: str) -> float:
        """文字列の類似度を計算する。類似度は0から1の間で1に近いほど類似度が高い。"""
        a_charset, b_charset = set(a), set(b)
        common_char_list = list(a_charset.union(b_charset))
        a_vector = [1 if c in a_charset else 0 for c in common_char_list]
        b_vector = [1 if c in b_charset else 0 for c in common_char_list]
        return self.cosine(a_vector, b_vector)
        
def main():
    # 悩みのデータ読み込み
    worry_text_data_dir = '../worry_text_data/user_text_data/user_text_data_one_sentence.csv'
    df_worry_text_data = pd.read_csv(worry_text_data_dir)

    # 名言のデータの読み込み
    famous_quote_data_dir = '../famous_quote_data/final_famous_quote_data/famous_quote_data.csv'
    df_famous_quote_data = pd.read_csv(famous_quote_data_dir)

    # 紐付け
    worry_text_list = df_worry_text_data['message'].to_list()
    famous_quote_list = df_famous_quote_data['famous_quote'].to_list()

    # famous_quote_link_listが最終的なoutput
    famous_quote_link_list = []

    cosine = CosineSimilarity()

    for worry_text in tqdm(worry_text_list):
        max_num = 0
        link_famous_quote = ''
        for famous_quote in famous_quote_list:
            cosine_similary  = cosine(worry_text, famous_quote)
            if cosine_similary > max_num:
                link_famous_quote = famous_quote
                max_num = cosine_similary
            
        
        famous_quote_link_list.append(link_famous_quote)
    
    
    # データフレーム化
    df_link = pd.DataFrame(list(zip(worry_text_list, famous_quote_link_list)), columns = ['worry_text', 'famous_quote'])
    df_link.to_csv('result/linked_worry_text_famous_quote.csv', index=False)
                



if __name__=='__main__':
    main()