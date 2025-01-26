from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
import torch
import os
from tqdm import tqdm
import numpy as np
import time
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# 首先加载所有音频列表
# root_path = '../data/General_Data/TikTok_Data/aural/'
root_path = '../data/weibo/aural/'
aural_path_list = os.listdir(root_path)

# Loading the audio file 使用librosa库加载音频文件，并提到音频片段大小为16000 Hz。它将音频片段转换成数组，并存储在“audio”变量中。
# load model and processor 导入预训练的Wav2Vec模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

# 循环每个音频文件，获取特征
aural_feature_list = []
# with open('../data/General_Data/TikTok_Data/aural_id_list.txt', 'w', encoding='utf-8') as f:
with open('../data/weibo/aural_id_list.txt', 'w', encoding='utf-8') as f:
    for aural_path in tqdm(aural_path_list):
        v_id = aural_path.split('/')[-1].split('.')[0]
        audio, rate = librosa.load(root_path + aural_path, sr=16000)
        # 下一步是获取输入值，将音频(数组)传递到分词器(tokenizer)，我们希望tensor是采用PyTorch格式，
        # 而不是Python整数格式。return_tensors =“pt”，这就是PyTorch格式。
        input_values = processor(audio, return_tensors="pt", padding="longest").input_values.to(device)  # Batch size 1
        with torch.no_grad():
            output = model(input_values).last_hidden_state
            feature = torch.mean(output, dim=1).cpu().numpy()[0]
            # 存储结果
            aural_feature_list.append(feature)
            f.write(f'{v_id}\n')
    # 结束之后将特征存储为numpy,[aural_num， 768]
    # np.save('../data/General_Data/TikTok_Data/aural_feature', np.array(aural_feature_list))
    np.save('../data/weibo/aural_feature', np.array(aural_feature_list))