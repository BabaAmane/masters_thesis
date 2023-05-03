import pandas as pd

def main():
    col_name = ["session_id","message_type","user_id","counsellor_id","message","message_id","posted_at"]
    # 元データ読み込み
    original_df = pd.read_csv('data/original_dialog_data.csv', names = col_name, index_col=0)
    # 処理によって多く追加されたカラムを削除

    original_df = original_df.drop(original_df.index[[0]])
    print('len original', len(original_df))

    # messageがNULL削除
    original_df = original_df.dropna(how='any')

    print('len null落とした', len(original_df))
    # いらないカラム削除
    dialog_df = original_df.drop(['message_id', 'posted_at'], axis=1)

    print('len dialog_df', len(original_df))
    print(dialog_df.isnull().sum())

    dialog_df.to_csv('after_ processing_data/dialog_data_after_process1.csv')

if __name__=='__main__':
    main()