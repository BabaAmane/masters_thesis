from transformers import T5Tokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import pandas as pd
from tqdm import tqdm
import time

def generate_reply(tokenizer,inp,model, device, max_len):
    input_text = "<s>" + str(inp) + "[SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    # 変数の修正が必要?
    max_len = int(max_len)
    out = model.generate(input_ids, 
                         do_sample=True, 
                         max_length=max_len, 
                         num_return_sequences=1, 
                         top_p=0.95, 
                         top_k=50, 
                         bad_words_ids=[[1], [5]], 
                         no_repeat_ngram_size=3)

    # print(">", "あなた")
    # print(inp)
    # print(">", "対話システム")
    
    encouragement_text = ''

    for sent in tokenizer.batch_decode(out):
        sent = sent.split('[SEP]</s>')[1]
        sent = sent.replace('</s>', '')
        sent = sent.replace('<br>', '\n')
        # print(sent)
        encouragement_text += sent
    
    return encouragement_text

    
def remove_character_from_list(lst, character):
    return [item.replace(character, '') for item in lst]


def main(args):
    pretrain_model = args.pretrain_model
    fainching_model = args.fainching_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(pretrain_model)
    tokenizer.do_lower_case = True

    print('fainching modelの読み込み ここから30min')
    model = AutoModelForCausalLM.from_pretrained(fainching_model)
    model.to(device)
    model.eval()
    print('fainching modelの読み込み 完了')


    # input_text_list = ['<s>後悔ばかりです。</s>', '<s>彼氏ができません。</s>', '<s>自分の恋愛運を信じれません。</s>', '<s>元カレに裏切られたのでこわいです。</s>', '<s>このまま諦めるのは嫌です。</s>', '<s>切なくて胸が避けそうです。</s>', '<s>ショックのあまり立ち直れないです。</s>', '<s>孤独になるのが怖い。</s>', '<s>就活の終わりが見えていない。</s>', '<s>意気消沈気味です。</s>', '<s>諦めた方がいいですか。</s>']

    # 評価データの読み込み
    rate = args.rate
    random_state = args.random_state
    test_dir = 'train_val/doc2vec/0824/test_' + str(rate) + '_' + str(random_state) + '.txt'
    lines = []
    with open(test_dir, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    input_text_list = []

    for item in lines:
        text_up_to_sep = item.split('[SEP]')[0]
        input_text_list.append(text_up_to_sep)
        
    # 可視化
    max_len = args.max_len
    encouragement_text_list = []
    for text in tqdm(input_text_list):
        # max len を40,60にする
        encouragement_text = generate_reply(tokenizer,text,model, device, max_len)
        encouragement_text_list.append(encouragement_text)

    # <s>, </s>の削除
    input_text_list = remove_character_from_list(input_text_list, '<s>')
    input_text_list = remove_character_from_list(input_text_list, '</s>')

    encouragement_text_list = remove_character_from_list(encouragement_text_list, '<s>')
    encouragement_text_list = remove_character_from_list(encouragement_text_list, '</s>')

    df_result = pd.DataFrame(list(zip(input_text_list, encouragement_text_list)), columns = ['input_text', 'encouragement_text'])
    result_dir = args.result_dir
    df_result.to_csv(result_dir, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', type=str, default='82',
                        help='rate')
    parser.add_argument('--random_state', type=str, default='42',
                        help='random_state')
    parser.add_argument('--max_len', type=str, default='60',
                        help='max len')
    parser.add_argument('--pretrain_model', type=str, default='rinna/japanese-gpt2-medium',
                        help='pretrain model name')
    parser.add_argument('--fainching_model', type=str, default='model/split_data_model/0824/medium_82_42_100epoch/checkpoint-12232',
                        help='fainching model name')
    parser.add_argument('--result_dir', type=str, default='reply_result/0824/result_82_42_11epoch.csv',
                        help='result_dir')
    
    


    args = parser.parse_args()
    main(args) 