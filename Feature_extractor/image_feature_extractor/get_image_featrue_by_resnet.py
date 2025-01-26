import cv2
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import os

root = '../data/weibo/videos/'
first_image_path = 'first_image.jpg'
video_path_list = os.listdir(root)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
v_model = create_feature_extractor(net, {'avgpool': 'feature'})  # 获取512维度的特征

image_feature_list = []

with open('../data/weibo/image_id_list.txt', 'w', encoding='utf-8') as f:
    for video_path in tqdm(video_path_list):
        v_id = video_path.split('/')[-1].split('.')[0]
        videoCapture = cv2.VideoCapture(root+video_path)
        success, frame = videoCapture.read()
        if not success:
            continue

        cv2.imwrite(first_image_path, frame)
        videoCapture.release()
        image_array = Image.open(first_image_path).convert('RGB')  # 获取加载后的原始图像
        image_tensor = image_transform(image_array).unsqueeze(0)

        with torch.no_grad():
            feature = v_model(image_tensor)['feature'].squeeze().cpu().numpy()
            image_feature_list.append(feature)
            f.write(f'{v_id}\n')
            f.flush()
    np.save('../data/weibo/image_feature', np.array(image_feature_list))