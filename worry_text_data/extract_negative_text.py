from asari.api import Sonar
import pandas as pd
from tqdm import tqdm

def main():
    dialog_df = pd.read_csv('user_text_data/user_text_data_all_sentence.csv')

    sonar = Sonar()

    text_list = list(dialog_df['message'])
    num_list = list(dialog_df['session_id'])
    negative_message_list = []
    negative_num_list = []

    for i, t in tqdm(zip(num_list, text_list)):
        res = sonar.ping(text=t)
        # print(i, res['top_class'])
        if res['top_class'] == 'negative':
            negative_message_list.append(t)
            negative_num_list.append(i)
            
    print(negative_message_list, negative_num_list)


    df = pd.DataFrame(list(zip(negative_num_list, negative_message_list)), columns = ['session_id', 'message'])
    df.to_csv('user_text_data_negative_judgment/user_text_data_negative_judgment.csv', index=False)





    

    

if __name__=='__main__':
    main()