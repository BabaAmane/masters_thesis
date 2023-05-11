import pandas as pd
import argparse

def main(args):
    #エクセルファイルの読み込み
    file_name = args.input_link_file
    print(file_name)
    input_df = pd.read_csv(file_name)
    

    print('処理前の長さ', len(input_df))
    input_df = input_df.dropna(how='any')
    print('drop nan', len(input_df))

    input_df.to_csv('linked_worry_text_famous_quote.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_link_file', type=str, default='linked_worry_text_famous_quote.csv',
                        help='名言と励まし文のデータ')

    args = parser.parse_args()
    main(args) 