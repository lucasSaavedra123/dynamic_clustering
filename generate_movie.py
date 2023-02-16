from moviepy.editor import *
from pathlib import Path

from PIL import Image
import PIL
import os
import glob

#accessing path of each image
for image_string in os.listdir('images/'):
    if image_string.endswith(".jpg"):
        base_width = 500
        image = Image.open(os.path.join('images/', image_string))
        width_percent = (base_width / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(width_percent)))
        image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
        image.save(os.path.join('low_images/', image_string))

img_clips = []
path_list=[]
#accessing path of each image
for image in os.listdir('low_images/'):
    if image.endswith(".jpg"):
        path_list.append(os.path.join('low_images/', image))
#creating slide for each image
for img_path in path_list:
  slide = ImageClip(img_path,duration=0.1)
  img_clips.append(slide)

video_slides = concatenate_videoclips(img_clips, method='compose')
video_slides.write_videofile("output_video.mp4", fps=10)
