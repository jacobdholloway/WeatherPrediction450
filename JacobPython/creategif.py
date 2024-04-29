from PIL import Image
import os

def create_gif_pillow(input_folder, output_file, duration=500):
    """
    Create a GIF from a sequence of images in a folder using Pillow.

    :param input_folder: str, path to the directory containing PNG images.
    :param output_file: str, path where the GIF should be saved.
    :param duration: int, duration in milliseconds between frames.
    """
    frames = []
    # Collect file names of PNGs in the directory
    file_names = sorted((fn for fn in os.listdir(input_folder) if fn.endswith('.png')), key=lambda x: int(x.split('.')[0]))
    # Load images
    for filename in file_names:
        frame = Image.open(os.path.join(input_folder, filename))
        frames.append(frame.copy())
        frame.close()
    # Save into a GIF file
    frames[0].save(output_file, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)

# Usage
create_gif_pillow('output', 'output_animation.gif', duration=500)
