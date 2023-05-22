from asari.api import Sonar
import pandas as pd
from tqdm import tqdm
from transformers import pipeline 

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    # 使うデータの処理
    dialog_df = pd.read_csv('user_text_data/user_text_data_all_sentence.csv')

    text_list = list(dialog_df['message'])
    num_list = list(dialog_df['session_id'])

    # # transformer 感情分析
    tokenizer = AutoTokenizer.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
   
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    negative_message_transformer_list = []
    negative_num_transformer_list = []
    negative_score_transformer_list = []

    # text_list = text_list[:200]
    for i, t in tqdm(zip(num_list, text_list)):
        if len(t) <= 512:
            res = sentiment_analyzer(t)
            # 条件狭めてもいいかも 70以上とか
            if res[0]['label'] == 'NEGATIVE':
                negative_message_transformer_list.append(t)
                negative_num_transformer_list.append(i)

                if res[0]['score'] < 0.5:
                    negative_score_transformer_list.append(0)
                elif res[0]['score'] >= 0.5 and res[0]['score'] < 0.6:
                    negative_score_transformer_list.append(5)
                elif res[0]['score'] >= 0.6 and res[0]['score'] < 0.7:
                    negative_score_transformer_list.append(6)
                elif res[0]['score'] >= 0.7 and res[0]['score'] < 0.8:
                    negative_score_transformer_list.append(7)
                elif res[0]['score'] >= 0.8 and res[0]['score'] < 0.9:
                    negative_score_transformer_list.append(8)
                else:
                    negative_score_transformer_list.append(9)

    # 学習用全体のデータ
    df_transformer = pd.DataFrame(list(zip(negative_num_transformer_list, negative_message_transformer_list)), columns = ['session_id', 'message'])
    df_transformer.to_csv('user_text_data_negative_judgment/user_text_data_negative_judgment_transformer.csv', index=False)

    # 分析用のデータ
    df_transformer_analysis = pd.DataFrame(list(zip(negative_num_transformer_list, negative_message_transformer_list, negative_score_transformer_list)), columns = ['session_id', 'message', 'score'])
    df_transformer_analysis.to_csv('analysis/user_text_data_negative_judgment_transformer_analysis.csv', index=False)
    
    

    # sonar = Sonar()
    # negative_message_asari_list = []
    # negative_num_asari_list = []
    # negative_score_asari_list = []
    # print('//////////////asari////////////////')
    # for i, t in tqdm(zip(num_list, text_list)):
    #     res = sonar.ping(text=t)
    #     #条件せば目ても良い
    #     if res['top_class'] == 'negative':
    #         negative_message_asari_list.append(t)
    #         negative_num_asari_list.append(i)

    #         if res['classes'][0]['confidence'] < 0.5:
    #             negative_score_asari_list.append(0)
    #         elif res['classes'][0]['confidence'] >= 0.5 and res['classes'][0]['confidence'] < 0.6:
    #             negative_score_asari_list.append(5)
    #         elif res['classes'][0]['confidence'] >= 0.6 and res['classes'][0]['confidence'] < 0.7:
    #             negative_score_asari_list.append(6)
    #         elif res['classes'][0]['confidence'] >= 0.7 and res['classes'][0]['confidence'] < 0.8:
    #             negative_score_asari_list.append(7)
    #         elif res['classes'][0]['confidence'] >= 0.8 and res['classes'][0]['confidence'] < 0.9:
    #             negative_score_asari_list.append(8)
    #         else:
    #             negative_score_asari_list.append(9)

    # # 学習用
    # df_asari = pd.DataFrame(list(zip(negative_num_asari_list, negative_message_asari_list)), columns = ['session_id', 'message'])
    # df_asari.to_csv('user_text_data_negative_judgment/user_text_data_negative_judgment_asari.csv', index=False)

    # # 分析用
    # df_asari_analysis = pd.DataFrame(list(zip(negative_num_asari_list, negative_message_asari_list, negative_score_asari_list)), columns = ['session_id', 'message', 'score'])
    # df_asari_analysis.to_csv('analysis/user_text_data_negative_judgment_asari.csv', index=False)



if __name__=='__main__':
    main()