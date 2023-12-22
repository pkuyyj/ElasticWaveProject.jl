from PIL import Image
import glob

# Path to the directory containing png files
path_to_png_files = './viz3D_out/stress*.png'

# Retrieve all png files in the directory
png_files = sorted(glob.glob(path_to_png_files))

# Create a list to store images
images = []

# Open each image and append it to the list
for filename in png_files:
    images.append(Image.open(filename))

# Create a GIF from the images
output_gif_path = './docs/wave_3D_contour.gif'
images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=True, duration=100, loop=0)


from PIL import Image
import glob
import cv2
import numpy as np

first_image = cv2.imread(png_files[0])
height, width, layers = first_image.shape

# Define the codec and create VideoWriter object
output_video_path = './docs/wave_3D_contour.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

# Iterate through each file, read the image, and add it to the video
for filename in png_files:
    img = cv2.imread(filename)
    video.write(img)

# Release the video writer
video.release()