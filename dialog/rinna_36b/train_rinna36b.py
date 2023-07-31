from transformers import AutoTokenizer
import argparse
import re
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
import transformers

# 参考にしたサイト
# https://note.com/npaka/n/nc387b639e50e

# 基本パラメータ
model_name = "rinna/japanese-gpt-neox-3.6b"

CUTOFF_LEN = 256  # コンテキスト長

# トークナイズ
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }

# プロンプトテンプレートの準備
def generate_prompt(data_point):
    # instruction消した
    if data_point["input"]:
        result = f"""### 指示:

### 入力:
{data_point["input"]}

### 回答:
{data_point["output"]}"""
    else:
        result = f"""### 指示:

### 回答:
{data_point["output"]}"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result


def remove_symbol(text: str)-> str:
    """記号を消す関数"""
    puncts = [',', '.', '"', ':', '!', '?', '|', ';', "'", '$', '&', '/', '>', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
                       '·',  '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',
                       '“', '★', '”', '●', 'â', '►',  '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '¥', '▓', '‹', '\u3000', '\u202f',
                       '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
                       '∙',  '↓', '、', '│', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '～',
                       '➡', '⇒', '▶', '「', '➄', '➆',  '➊', '➋', '➌', '➍', '⓪', '①', '②', '③', '④', '⑤', '⑰', '❶', '❷', '❸', '❹', '❺', '❻', '❼', '❽',  
                       '＝', '※', '㈱', '､', '△', '℮', 'ⅼ', '‐', '┝', '↳', '◉', '／', '＋', '○',
                        '✅', '☑', '➤', 'ﾞ', '↳', '〶', '☛', '⁺', '『', '≫', 'ゝ', 'ゝ' ,'゛', '：', '〒', '∥', '￣', '゜', '＼', '‥', '℃','〃', '〆',
                       '+', '※', '：', '＠', '＊', '──', '．', '？', '(', ')', '（' ,'）', '[', ']', '{', '}', '【', '】', '〔', '〕', '〈', '〉', '『', '』', '《', '》',
                       '<', '>', '❤️', '〜', '♪', 'θ', 'ノ', '´', '｀', '￣','…', '％', '%', '-', '_', 'ﾟ▽ﾟ', '」', '」', '☌ᴗ☌', '･･', '･ω･', '[', ']', '「', '」', '(', ')'
                       ] 

    for punct in puncts:
        text = text.replace(punct, '')
        
    return text

def remove_alphabet(text: str)-> str:
    """アルファベットを削除する関数"""
    text = re.sub(r'[a-zA-Z]+', '', text)
    text = re.sub(r'[ａ-ｚ Ａ-Ｚ]+', '', text)
    return text

def make_input_output_list(file_name: str, rate: str, random_state: str)-> list:
    dir = '../train_val/doc2vec/' + file_name + '_' + str(rate) + '_' + str(random_state) + '.txt'
    lines = []
    with open(dir, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())

    input_list = []
    for item in lines[:3]:
        text_up_to_sep = item.split('[SEP]')[0]
        input_list.append(text_up_to_sep)

    output_list = []
    for text in lines[:3]:
        sep_index = text.index("[SEP]")  # [SEP]のインデックスを取得
        extracted_text = text[sep_index + 5:-4]  # [SEP]の後ろのテキストを抽出（[SEP]自体の長さが5、</s>の長さが4）
        output_list.append(extracted_text)
    
    return input_list, output_list

def make_dataset(input_list: list, output_list: list)-> list:
    data = []
    for input_text, output_text in zip(input_list, output_list):
        input_text = remove_alphabet(input_text)
        input_text = remove_symbol(input_text)
        data.append({'input': input_text, 'output': output_text})
    return data

def main(args):
    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # trainを読み込んでinputとoutputの辞書で分ける
    
    rate = args.rate
    random_state = args.random_state
    train_input_list, train_output_list = make_input_output_list('train', rate, random_state)
    val_input_list, val_output_list = make_input_output_list('val', rate, random_state)
    
    # 辞書の作成
    train_data = make_dataset(train_input_list, train_output_list)
    val_data = make_dataset(val_input_list, val_output_list)

    # 学習
    # モデルの準備
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
    )

    # LoRAのパラメータ
    lora_config = LoraConfig(
        r= 8, 
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # モデルの前処理
    model = prepare_model_for_int8_training(model)

    # LoRAモデルの準備
    model = get_peft_model(model, lora_config)

    # 学習可能パラメータの確認
    model.print_trainable_parameters()

    eval_steps = 200
    save_steps = 200
    logging_steps = 20
    epoch = args.epoch

    output_dir = "result/lora-rinna-3.6b_" + rate + '_' + random_state + '_'+ str(epoch)

    # トレーナーの準備
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            num_train_epochs=epoch,
            learning_rate=3e-4,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            output_dir=output_dir,
            report_to="none",
            save_total_limit=3,
            push_to_hub=False,
            auto_find_batch_size=True
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 学習の実行
    model.config.use_cache = False
    trainer.train() 
    model.config.use_cache = True

    peft_name = "model/lora-rinna-3.6b_" + rate + '_' + random_state + '_'+ str(epoch)
    # LoRAモデルの保存
    trainer.model.save_pretrained(peft_name)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rate', type=str, default='82',
                        help='rate')
    parser.add_argument('--random_state', type=str, default='42',
                        help='random_state')
    parser.add_argument('--result_dir', type=str, default='eval_result/82_42.csv',
                        help='結果')
    parser.add_argument('--epoch', type=int, default=3,
                        help='結果')

    args = parser.parse_args()
    main(args) 