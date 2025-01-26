from moviepy.editor import *
from tqdm import tqdm
import cv2
# 提取音频信息

# 获取视频地址列表
# root = '../data/Douyin_Data/videos/'
# root_path = '../data/General_Data/TikTok_Data/videos/'
root_path = '../data/weibo/videos/'
video_path_list = os.listdir(root_path)

# out_path = '../data/General_Data/TikTok_Data/aural/'
out_path = '../data/weibo/aural/'

# 获取有comment的video列表
# selected_video_ids = [video_id.strip() for video_id in open('../data/Tiktok_Data/selected_videos.txt', 'r')]

# 循环获取音频数据
for video_path in tqdm(video_path_list):

    # 加载视频数据
    v_id = video_path.split('/')[-1].split('.')[0]
    # if v_id not in selected_video_ids:
    #     continue
    # 首先获取视频总帧数
    capture = cv2.VideoCapture(root_path + video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    time_count = frame_count / 30  # 计算视频时间,取整数
    # 如果视频太长，大于5分钟的话，可能有问题，直接跳过
    if time_count > 300 or time_count < 2:
        continue
    audio = VideoFileClip(root_path + video_path).audio
    audio.write_audiofile(out_path + f'{v_id}.wav')