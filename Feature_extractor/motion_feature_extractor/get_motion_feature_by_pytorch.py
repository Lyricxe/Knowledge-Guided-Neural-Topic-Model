from torchvision.io.video import read_video
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.video import r3d_18, R3D_18_Weights
import os
import torch
from tqdm import tqdm
import numpy as np
import cv2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
weights = R3D_18_Weights.DEFAULT
preprocess = weights.transforms()
net = r3d_18(weights=weights)
model = create_feature_extractor(net, {'avgpool': 'feature'})
model.eval().to(device)

path = '../data/weibo/videos/'
video_path_list = os.listdir(path)

motion_feature_list = []
with open('../data/weibo/motion_id_list.txt', 'w', encoding='utf-8') as f:
    time_count_list = []
    for video_path in tqdm(video_path_list):
        v_id = video_path.split('/')[-1].split('.')[0]
        print(f'v_id:{v_id}')
        capture = cv2.VideoCapture(path+video_path)
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        time_count = frame_count/30
        capture.release()
        time_count_list.append(time_count)

        for start in np.arange(0, time_count, 0.5):
            batch_data, _, _ = read_video(path + video_path, start_pts=start, end_pts=start+0.033, pts_unit='sec',
                                      output_format="TCHW")
            if batch_data.shape[0] == 0:
                continue
            if start == 0:
                v_data = batch_data
            else:
                v_data = torch.cat((v_data, batch_data), dim=0)
        batch = preprocess(v_data).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(batch)['feature'].squeeze().cpu().numpy()
            motion_feature_list.append(feature)
            f.write(f'{v_id}\n')
    np.save('../data/weibo/motion_feature', np.array(motion_feature_list))