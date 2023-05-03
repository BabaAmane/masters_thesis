import pandas as pd

def main():
    # userの余計な文章削除
    dialog_df = pd.read_csv('after_ processing_data/dialog_data_after_process2.csv')

    # userだけ抽出
    user_text_df = dialog_df[dialog_df['message_type'] == 'user']


    # 以降いらないテキスト情報を削除していく
    # 文字列が短いものを削除 1は削除、0は消さない
    # 特定の文字列を含む場合削除
    drop_list = []
    drop_reply_list = ['こんにちは', 'よろしくお願いします','ありがとうございます', 'そうなのですか', 'なるほどわかりました', \
                      'どうぞよろしくお願いします', 'よろしくお願いいたします' 'おっしゃる通りです', 'なるほどありがとうございます', 'それはそうですね', \
                      'ごもっともですね', 'なんでわかるんですか', 'わかりました', 'そうなんですか', 'お願いいたします','お願いします', 'ふむふむなるほど', \
                      'そうなんですね', 'なるほど', '初めましてです', 'ありがとうございました', 'それはないですね', 'そーなんですよ', 'それはないですね', 'これでよいでしょうか', \
                      'なるほど楽しみです', 'おおそうなんですね', 'なるほど大変参考になります', 'はじめまして', 'そうなんですね', 'ええー', 'なんですか',  \
                      'ほかのも聞きたいです', 'すみません', 'よろしくお願い致します'
                      
                      ]
    drop_spesific_word_list = ['利き手','生年月日','年月日', '見ていただけますか', '手相', '誕生日', '座', '型', '当たっております']
    drop_word_list = drop_reply_list + drop_spesific_word_list
    print(drop_word_list)

    # めかぶかなんかで人の名前だった場合削除

    for messsage in user_text_df['message']:
        if len(messsage) > 7:
            i = 0
            for drop_word in drop_word_list:
                index = messsage.find(drop_word)
                if index != -1:
                    # あまりにも長すぎる場合は悩みの可能性が高いので処理が必要
                    if len(messsage) > 30:
                        drop_list.append(0)
                        break
                    else:
                        drop_list.append(1)
                    break
                if i == len(drop_word_list) - 1 :
                    drop_list.append(0)
                    break
                i += 1
        else:
            drop_list.append(1)
    
    # データフレームにカラムを追加
    user_text_df['drop_flag'] = drop_list

    # dropが1のデータは削除
    user_text_df = user_text_df[user_text_df['drop_flag'] == 0]

    # いらないカラム削除
    user_text_df.drop(['message_type', 'user_id', 'counsellor_id', 'drop_flag'], axis=1, inplace=True)
    # 前処理を一旦終えたファイル
    user_text_df.to_csv('user_text_data/user_text_data_all_sentence.csv', index=False)


    ## 会話の一番初めのデータだけを取得
    # session_idの取得
    # session_id_list = user_text_df['session_id'].to_list()
    # session_id_list = sorted(set(session_id_list), key=session_id_list.index)

    user_text_df.drop_duplicates(subset='session_id', keep='first', inplace=True)
    user_text_df.to_csv('user_text_data/user_text_data_one_sentence.csv', index=False)

   


if __name__ == '__main__':
    main()