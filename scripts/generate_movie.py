from pathlib import Path
from PIL import Image
import PIL
import os
import glob

from moviepy.editor import *


IMAGES_PATHS = [F'./experiment_{i}_plots' for i in range(0,1)]

for image_path in IMAGES_PATHS:

  for image_string in os.listdir(image_path):
      if image_string.endswith(".jpg"):
          base_width = 1000
          image = Image.open(os.path.join(image_path, image_string))
          width_percent = (base_width / float(image.size[0]))
          hsize = int((float(image.size[1]) * float(width_percent)))
          image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
          image.save(os.path.join('_low_images/', image_string))

  img_clips = []
  path_list=[]
  #accessing path of each image
  for image in os.listdir('_low_images/'):
      if image.endswith(".jpg"):
          path_list.append(os.path.join('_low_images/', image))
  #creating slide for each image
  for img_path in path_list:
    slide = ImageClip(img_path,duration=0.1)
    img_clips.append(slide)

  video_slides = concatenate_videoclips(img_clips, method='compose')
  video_slides.write_videofile(os.path.join(image_path, "output_video.mp4"), fps=10)
