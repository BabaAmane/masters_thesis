import pandas as pd

# 余計なデータを削除するファイル

def contains_word_japanese(text: str, drop_word_list: list):
    # 文章を単語や句読点などで分割し、単語ごとに比較する
    words = text.split()
    for w in words:
        for drop_word in drop_word_list:
            if drop_word in w:
                # print('drop_word', drop_word)
                return True
    return False

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
                      'ほかのも聞きたいです', 'すみません', 'よろしくお願い致します', 'ありがとうございます'
                      
                      ]
    drop_spesific_word_list = ['利き手','生年月日','年月日', '見ていただけますか', '手相', '誕生日', '座', '型', '当たっております']
    drop_word_list = drop_reply_list + drop_spesific_word_list
    # print(drop_word_list)


    for messsage in user_text_df['message']:
        if len(messsage) > 7:
            judgment = contains_word_japanese(messsage, drop_word_list)
            if judgment == True:
                drop_list.append(1)
                # print('除去対象悩みの文章')
                # print(messsage)
            else:
                drop_list.append(0)
    
        else:
            drop_list.append(1)
    
    print(len(user_text_df))
    print(len(drop_list))

    session_id_list = list(user_text_df['session_id'])
    message_list = list(user_text_df['message'])

    df_result = pd.DataFrame(list(zip(session_id_list, message_list, drop_list)), columns = ['session_id', 'message', 'drop_flag'])
    
    # データフレームにカラムを追加
    # user_text_df['drop_flag'] = drop_list
    # user_text_df.loc[:, 'drop_flag'] = drop_list

    # dropが1のデータは削除
    # user_text_df = user_text_df[user_text_df['drop_flag'] == 0]
    df_result = df_result[df_result['drop_flag'] == 0]
    # いらないカラム削除s
    # user_text_df.drop(['message_type', 'user_id', 'counsellor_id', 'drop_flag'], axis=1, inplace=True)
    # 前処理を一旦終えたファイル
    # user_text_df.to_csv('user_text_data/user_text_data_all_sentence.csv', index=False)
    df_result.to_csv('user_text_data/user_text_data_all_sentence.csv', index=False)

    ## 会話の一番初めのデータだけを取得
    
    # user_text_df.drop_duplicates(subset='session_id', keep='first', inplace=True)
    # user_text_df.to_csv('user_text_data/user_text_data_one_sentence.csv', index=False)

   


if __name__ == '__main__':
    main()