import pandas as pd
import re

def remove_alphabet(text: str)-> str:
    """アルファベットを削除する関数"""
    text = re.sub(r'[a-zA-Z]+', '', text)
    text = re.sub(r'[ａ-ｚ Ａ-Ｚ]+', '', text)
    return text

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
                       '<', '>', '❤️', '〜', '♪', 'θ', 'ノ', '´', '｀', '￣','…', '％', '%', '-', '_', 'ﾟ▽ﾟ'
                       
                       ] 

    for punct in puncts:
        text = text.replace(punct, '')
        
    return text

def chenge_pronunciation(text: str) -> str:
    puncts = '！'
    text = re.sub(puncts, '。', text)
    return text

def remove_number(text: str) -> str:
    puncts = r'[0-9]'
    text = re.sub(puncts, '', text)
    return text


def main():
    
    original_df = pd.read_csv('after_ processing_data/dialog_data_after_process1.csv', index_col=0)
    
    print('len original', len(original_df))

    # messageに記号が入いっていた場合削除

    original_df['message'] = original_df['message'].map(remove_symbol)
    original_df['message'] = original_df['message'].map(remove_alphabet)
    original_df['message'] = original_df['message'].map(remove_number)

    # messageがNULL削除
    original_df = original_df.dropna(how='any')

    original_df = original_df[original_df['message'] != '']
    original_df = original_df[original_df['message'] != ' ']


    print('len 記号削除＋null削除', len(original_df))
    
    print(original_df.isnull().sum())

    original_df.to_csv('after_ processing_data/dialog_data_after_process2.csv')

if __name__=='__main__':
    main()