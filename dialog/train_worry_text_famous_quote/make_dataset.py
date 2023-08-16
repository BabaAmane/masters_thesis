import pandas as pd
from transformers import AutoModelForCausalLM, T5Tokenizer
import argparse
from tqdm import tqdm

def main(args):
    #エクセルファイルの読み込み
    file_name = args.input_link_file
    input_df = pd.read_csv(file_name)

    print(input_df.columns)

    input_text_list = input_df['worry_text'].to_list()
    output_text_list = input_df['famous_quote'].to_list()

    
    print('input_text_list len', len(input_text_list))
    print('output_text_list len', len(output_text_list))

    print(input_text_list[0])

    # 全てテキストファイルの作成
    # pretrain_model = args.pretrain_model
    # tokenizer = T5Tokenizer.from_pretrained(pretrain_model)

    tmp = []
    output_file_dir = args.output_file
    output_file = open(output_file_dir, 'w', encoding='utf-8')


    for i in tqdm(range(len(input_text_list))):
        # 修正後のコード
        # inp_tokens = tokenizer.tokenize(input_text_list[i])[:256]
        # inp = "".join(inp_tokens).replace('▁', '')
        # # print(output_text_list[i])
        # out_tokens = tokenizer.tokenize(output_text_list[i])[:256]
        # out = "".join(out_tokens).replace('▁', '')

        # data = "<s>" + inp + "[SEP]" + out + "</s>"
        # tmp.append(data)

        # 修正前のコード
        data = "<s>" + str(input_text_list[i]) + "[SEP]" + str(output_text_list[i]) + "</s>" +'\n'
        tmp.append(data)
        output_file.write(data)
    
    # txt = "".join(tmp) 
    # output_file.write(txt+ '\n')
    
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default='rinna/japanese-gpt2-medium',
                        help='pretrain model name')
    parser.add_argument('--input_link_file', type=str, default='../../link_famous_quote_worry_text/result/linked_negative_text_famous_quote_doc2vec.csv',
                        help='名言と励まし文のデータ')
    parser.add_argument('--output_file', type=str, default='train_data_worry_text_famous_quote/dataset_negative_text_famous_quote2.txt',
                        help='出力のディレクトリ')
    
    args = parser.parse_args()
    main(args) 