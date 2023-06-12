from transformers import T5Tokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def generate_reply(tokenizer,inp,model, device, num_gen=1):
    input_text = "<s>" + str(inp) + "[SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    # 変数の修正が必要?
    out = model.generate(input_ids, 
                         do_sample=True, 
                         max_length=30, 
                         num_return_sequences=1, 
                         top_p=0.95, 
                         top_k=50, 
                         bad_words_ids=[[1], [5]], 
                         no_repeat_ngram_size=3)

    print(">", "あなた")
    print(inp)
    print(">", "対話システム")
    # print("out", out)
    # print("新規", tokenizer.decode(out.tolist()[0]))
    # print(tokenizer.batch_decode(out))

    for sent in tokenizer.batch_decode(out):
        sent = sent.split('[SEP]</s>')[1]
        sent = sent.replace('</s>', '')
        sent = sent.replace('<br>', '\n')
        print(sent)


def generate_reply_another(tokenizer,inp,model):  
    input_text = "<s>" + str(inp) + "[SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(model.device),
            do_sample=True,
            max_length=30, 
            top_p=0.95,
            top_k=100,
            num_return_sequences=1,
            max_new_tokens=100,
            bad_words_ids=[[1], [5]],
            no_repeat_ngram_size=1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0])
    sent = (
        output.replace(inp, "")
        .replace(" ", "")
        .replace("<s>", "")
        .replace("</s>", "")
        .replace("[SEP]", "")
    )

    # print('output', output)
    tex = ["。", "?", "？", "！", "!", "."]
    for t in tex:
        if t in sent:
            sent = sent.split(t)[0] + t
        else:
            sent = sent
        return sent



def main(args):
    pretrain_model = args.pretrain_model
    fainching_model = args.fainching_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(pretrain_model)
    tokenizer.do_lower_case = True

    model = AutoModelForCausalLM.from_pretrained(fainching_model)
    model.to(device)
    model.eval()

    msg = args.input_text
    text_list = ['後悔ばかりです。', '彼氏ができません。', '自分の恋愛運を信じれません。', '元カレに裏切られたのでこわいです。', 'このまま諦めるのは嫌です。', '切なくて胸が避けそうです。', 'ショックのあまり立ち直れないです。', '孤独になるのが怖い。', '就活の終わりが見えていない', '意気消沈気味です', '諦めた方がいいですか']
    
    # print('変更前')
    for text in text_list:
        generate_reply(tokenizer,text,model, device)
    # print('変更後')
    # output = generate_reply_another(tokenizer,msg,model)
    # print(output)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_model', type=str, default='rinna/japanese-gpt2-small',
                        help='pretrain model name')
    parser.add_argument('--fainching_model', type=str, default='model_negative_20epoch/',
                        help='fainching model name')
    parser.add_argument('--input_text', type=str, default='彼女ができません。',
                        help='input text')


    args = parser.parse_args()
    main(args) 