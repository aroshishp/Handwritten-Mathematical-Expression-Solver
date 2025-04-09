import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import albumentations as A
import cv2



# List of fonts (ensure these fonts are installed or provide paths to .ttf files)
# font_paths = ["arial.ttf", "times.ttf", "calibri.ttf"]  # Add more fonts if available
font_paths = [
    "C:/Windows/Fonts/BRADHITC.TTF",
    "C:/Windows/Fonts/BRUSHSCI.TTF",
    "C:/Windows/Fonts/Cookie-Regular.ttf",
    "C:/Windows/Fonts/Inkfree.ttf",
    "C:/Windows/Fonts/FREESCPT.ttf",
    "C:/Windows/Fonts/GreatVibes-Regular.ttf",
    "C:/Windows/Fonts/PAPYRUS.ttf",
    "C:/Windows/Fonts/Yellowtail-Regular.ttf",
]

font_paths_math = [
    # "C:/Windows/Fonts/Cookie-Regular.ttf" ,
    "C:/Windows/Fonts/Inkfree.ttf" ,
    "C:/Windows/Fonts/Cambriai.ttf" ,
    "C:/Windows/Fonts/times.ttf" ,
    "C:/Windows/Fonts/Arial.ttf" ,
    "C:/Windows/Fonts/Calibri.ttf" ,
    "C:/Windows/Fonts/Georgia.ttf" ,
    # "C:/Windows/Fonts/BAUHS93.ttf" ,
    # "C:/Windows/Fonts/ITCBLKAD.ttf" ,
    "C:/Windows/Fonts/Gabriola.ttf" ,
    # "C:/Windows/Fonts/FREESCPT.ttf" ,
    # "C:/Windows/Fonts/GreatVibes-Regular.ttf" ,
    # "C:/Windows/Fonts/PAPYRUS.ttf" ,
    # "C:/Windows/Fonts/Yellowtail-Regular.ttf" ,
    # "C:/Windows/Fonts/MISTRAL.ttf" ,
    # "C:/Windows/Fonts/CHILLER.ttf" ,
    # "C:/Windows/Fonts/CURLZ___.ttf" ,
    # "C:/Windows/Fonts/Kalam.ttf" ,
]

#t: -Inkfree
#e: -yellowtail

# Augmentation pipeline using Albumentations
augmentation = A.Compose([
    A.Rotate(limit=15, p=0.5),  # Random rotation
    A.GaussianBlur(blur_limit=(3,5), p=0.6),  # Slight blurring
    # A.GaussNoise(var_limit=(0, 0.1), p=0.5),  # Add noise
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.1, 0.5), rotate_limit=0, p=0.5),  # Scaling and slight shifts
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),  # Brightness/contrast adjustment
])

def generate_synthetic_images(character, num_samples, output_dir):
    """Generate synthetic images of a given character."""
    for i in range(num_samples):
        # Create a blank white image
        img_size = 64
        img = Image.new('L', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(img)

        # Choose a random font and font size
        font_path = random.choice(font_paths_math)
        font_size = random.randint(60, 70)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"Error: Font '{font_path}' not found. Please ensure the font file exists.")
            continue

        # Calculate text size and position it at the center
        text_bbox = draw.textbbox((0, 0), character, font=font)  # Get exact bounding box
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (img_size - text_width) // 2 - text_bbox[0]
        text_y = (img_size - text_height) // 2 - text_bbox[1]

        # Draw the character
        draw.text((text_x, text_y), character, fill='black', font=font)

        # Convert the image to a NumPy array for augmentation
        img_array = np.array(img)

        # Apply augmentation
        augmented = augmentation(image=img_array)
        augmented_img = augmented['image']

        # Save the augmented image
        cv2.imwrite(os.path.join(output_dir, f"pi_{i}.png"), augmented_img)

# characters = 'abcdeijkmnpqrstuvwxyz'
# characters = 'abcdijkmnpqrsuvwxyz'
# characters = '0123456789'

# characters = '+'
# characters = '-'
# characters = '='
# characters = '('
# characters = ')'
# characters = '>'
# characters = '<'
# characters = '÷'
# characters = '×'
# characters = '≤'
# characters = '≥'
# characters = '±'
# characters = 'ʃ'
# characters = '∞'
# characters = '―'
# characters = 'θ'
characters = 'π'

# Main Function
if __name__ == "__main__":
    num_samples = int(input("Enter the number of samples to generate: "))
    for character in characters:
        # Define the output directory
        output_dir = f"images/pi"
        os.makedirs(output_dir, exist_ok=True)
        generate_synthetic_images(character, num_samples, output_dir)
        print(f"Dataset generated in '{output_dir}'.")
