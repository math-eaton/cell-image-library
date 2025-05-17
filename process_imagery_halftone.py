from PIL import Image
import numpy as np
import os

# Process the image using Floyd-Steinberg error diffusion
def process_image(image, output_path, aspect_ratio=None, final_size=(960, 960)):
    from PIL import ImageEnhance

    min_resolution = 250
    width, height = image.size
    if width < min_resolution or height < min_resolution:
        print(f"Image for application_id {os.path.basename(image_path).split('.')[0]} was not processed due to low resolution.")
        return None

    if aspect_ratio:
        new_width = int(min(width, height) * aspect_ratio)
        new_height = int(min(width, height))
        image = image.resize((new_width, new_height), Image.BILINEAR)

    image = image.convert('L')
    
    # Apply halftone dithering
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    image = image.convert('1')

    image = image.convert('RGB')
    image = image.convert('RGBA')

    data = np.array(image)
    red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_areas = (red == 255) & (green == 255) & (blue == 255)
    data[white_areas] = [255, 255, 255, 0]
    image = Image.fromarray(data)

    image = image.resize(final_size, Image.NEAREST)
    print(f"rescaling to {final_size} pixels")

    if aspect_ratio:
        width, height = image.size
        crop_percentage = 2
        crop_pixels = int(min(width, height) * (crop_percentage / 100))

        left = crop_pixels
        top = crop_pixels
        right = width - crop_pixels
        bottom = height - crop_pixels

        image = image.crop((left, top, right, bottom))

    image.save(output_path, optimize=True)
    return image
