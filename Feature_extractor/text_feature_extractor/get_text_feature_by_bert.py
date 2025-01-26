from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pretrained_path = "bert-base-chinese"
# pretrained_path = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(pretrained_path)
Bert_model = BertModel.from_pretrained(pretrained_path).eval()
Bert_model.to(device)


def trans_feature_extractor(root_path):
    df_data = pd.read_csv(f'{root_path}/transcription/all_videos_transcript.csv', sep=',')
    trans_dict = {}
    for idx, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        v_id = row['video_id']
        text = str(row['text'])
        trans_dict[v_id] = text

    trans_feature_list = []
    with open(f'{root_path}/transcription/trans_id_list.txt', 'w', encoding='utf-8') as f:
        for v_id, text in tqdm(trans_dict.items()):
            if len(text) <= 10:
                continue
            inputs = bert_tokenizer([text], truncation=True, max_length=500, return_tensors="pt").to(device)
            with torch.no_grad():
                out = Bert_model(**inputs).last_hidden_state
                feature = torch.mean(out, dim=1).squeeze(0).cpu().numpy()
                trans_feature_list.append(feature)
                f.write(f'{v_id}\n')
        np.save(f'{root_path}/transcription/trans_feature', np.array(trans_feature_list))


def comment_featrue_extractor(root_path):

    df_data = pd.read_csv(f'{root_path}/comment/all_video_comments.txt', sep='\t')
    comment_dict = {}
    for idx, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        v_id = row['video_id']
        text = str(row['text'])
        comment_dict[v_id] = text

    comment_feature_list = []
    with open(f'{root_path}/comment/comment_id_list.txt', 'w', encoding='utf-8') as f:
        for v_id, text in tqdm(comment_dict.items()):
            inputs = bert_tokenizer([text], truncation=True, max_length=500, return_tensors="pt").to(device)
            with torch.no_grad():
                out = Bert_model(**inputs).last_hidden_state
                feature = torch.mean(out, dim=1).squeeze(0).cpu().numpy()
                comment_feature_list.append(feature)
                f.write(f'{v_id}\n')

        np.save(f'{root_path}/comment/comment_feature', np.array(comment_feature_list))


if __name__ == '__main__':
    root_path = '../data/weibo/'
    trans_feature_extractor(root_path)