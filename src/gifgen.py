from PIL import Image
import glob

# Path to the directory containing png files
path_to_png_files = './viz3D6k_out-damp80m500s4/T_3D*.png'

# Retrieve all png files in the directory
png_files = sorted(glob.glob(path_to_png_files))

# Create a list to store images
images = []

# Open each image and append it to the list
for filename in png_files:
    images.append(Image.open(filename))

# Create a GIF from the images
output_gif_path = './docs/wave_3D-damp80m500s4-log.gif'
images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=True, duration=100, loop=0)
