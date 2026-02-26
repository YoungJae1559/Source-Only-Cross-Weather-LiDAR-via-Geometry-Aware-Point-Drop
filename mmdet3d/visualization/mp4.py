import cv2
import os
from glob import glob
from natsort import natsorted
from tqdm import tqdm

input_folder = '/home/vip/harry/LiDARWeather/vis_results_first'
output_video_path = '/home/vip/harry/LiDARWeather/Original_diff_view.mp4'

image_files = natsorted(glob(os.path.join(input_folder, '*.png')))

if not image_files:
    raise ValueError("No PNG files found in the input folder.")

sample_img = cv2.imread(image_files[0])
height, width, _ = sample_img.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))

for img_path in tqdm(image_files, desc="Creating MP4", unit="frame"):
    img = cv2.imread(img_path)
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    video_writer.write(img)

video_writer.release()
print(f"\n MP4 video successfully saved to: {output_video_path}")