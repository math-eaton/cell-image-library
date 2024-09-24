import requests
import config
from PIL import Image
from PIL import ImageOps
from io import BytesIO
import os
import random
import time
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


# Record the start time
start_time = time.time()

try:
    with open('processed_images.txt', 'r') as file:
        processed_ids = set(line.strip() for line in file)
except FileNotFoundError:
    processed_ids = set()

# Define the number of images to download
num_images = 15

# Define the output folder
output_folder = "output/cinema"

# Base URL for the API
api_url = "https://cilia.crbs.ucsd.edu/rest"

# Authentication details
username = config.CIL_API_USER
password = config.CIL_API_PW

# Define the fields for CCDB images
ccdb_fields = [
    "CIL_CCDB.CCDB.Recon_Display_image.URL",
    "CIL_CCDB.CCDB.Image2d.Image2D_Display_image.URL",
    "CIL_CCDB.CCDB.Segmentation.Seg_Display_image.URL",
]

# Identify and crop any letterbox around the image
# higher sensitivity considers more grey values +/- 0 to 255 aka pure white/black
def crop_image(image, sensitivity=1):
    # Convert the image to a NumPy array
    image_data = np.array(image)

    if len(image_data.shape) == 3:  # RGB Image
        # Identify non-mono pixels
        non_white_black = np.any(image_data < (255 - sensitivity), axis=-1) & np.any(image_data > sensitivity, axis=-1)
    else:  # Grayscale Image
        non_white_black = (image_data < (255 - sensitivity)) & (image_data > sensitivity)

    # Get the bounding box of the non-mono pixels
    non_white_black_bounding_box = np.argwhere(non_white_black)

    # Crop the image to the bounding box
    cropped_image = image_data[non_white_black_bounding_box.min(axis=0)[0]:non_white_black_bounding_box.max(axis=0)[0] + 1,
                               non_white_black_bounding_box.min(axis=0)[1]:non_white_black_bounding_box.max(axis=0)[1] + 1]

    # Return the cropped image
    return Image.fromarray(cropped_image)

# Assess the qualities of an image before dithering
def calculate_brightness(image):
    try:
        grayscale = image.convert('L')
        histogram = grayscale.histogram()
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale

    except Exception as e:
        print(f"An error occurred in calculate_brightness: {str(e)}")
        return None

def calculate_contrast(image):
    try:
        grayscale = image.convert('L')
        grayscale_array = np.array(grayscale)
        contrast = grayscale_array.std()

        return contrast

    except Exception as e:
        print(f"An error occurred in calculate_contrast: {str(e)}")
        return None
    
def calculate_entropy(image):
    try:
        # Convert the image to grayscale
        grayscale = image.convert('L')
        
        # Calculate the histogram
        histogram = grayscale.histogram()

        # Normalize the histogram to get probabilities
        histogram_length = sum(histogram)
        probability_histogram = [float(h) / histogram_length for h in histogram]

        # Calculate entropy
        entropy = -sum([p * np.log2(p) for p in probability_histogram if p != 0])

        print("assessing image qualities...")
        return entropy

    except Exception as e:
        print(f"an error occurred in calculate_entropy: {str(e)}")
        return None


# Process the image using Floyd-Steinberg error diffusion
def process_image(image):
    
    # Check the input image resolution
    min_resolution = 144  # Set minimum resolution
    width, height = image.size
    if width < min_resolution or height < min_resolution:
        return None
    
    # Resize the image (pre-dither) while maintaining aspect ratio
    # Avoid forcing a fixed size here
    image.thumbnail((1920, 1920), Image.BILINEAR)  # Ensure a max size while preserving aspect ratio
    
    # Convert the image to grayscale
    image = image.convert('L')

    # Dither the image
    image = image.convert('1')
    print("Dithering...")

    # Convert the image back to RGB
    image = image.convert('RGB')

    # Make sure the image has an alpha channel
    image = image.convert('RGBA')

    # Convert white (also shades of whites) pixels to transparent
    data = np.array(image)
    red, green, blue, alpha = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_areas = (red > 200) & (green > 200) & (blue > 200)
    data[white_areas] = [255, 255, 255, 0]
    image = Image.fromarray(data)

    # Optionally crop the outer 2% (you can remove this if not needed)
    width, height = image.size
    left = width * 0.02
    top = height * 0.02
    right = width * 0.98
    bottom = height * 0.98
    image = image.crop((left, top, right, bottom))

    # Dynamically calculate the final output size while preserving aspect ratio
    final_width = width
    final_height = height

    # Resize dynamically if needed, remove hardcoded size
    max_width = 1920  # Limit to a maximum width
    if final_width > max_width:
        final_height = int((max_width / final_width) * final_height)
        final_width = max_width

    # Resize the image using the dynamically calculated size
    image = image.resize((final_width, final_height), Image.NEAREST)
    print(f"Rescaling to {final_width}x{final_height}...")

    return image
# Configure retries
retry_strategy = Retry(
    total=5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)


# Function to crop image to a specific aspect ratio
def crop_to_aspect_ratio(image, target_ratio):
    width, height = image.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        # The image is too wide, crop the width
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        cropped_image = image.crop((left, 0, right, height))
    elif current_ratio < target_ratio:
        # The image is too tall, crop the height
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        cropped_image = image.crop((0, top, width, bottom))
    else:
        # The image already matches the target ratio, no cropping needed
        cropped_image = image

    return cropped_image



def download_and_maybe_process_image(image_id, process=True, crop_ratio=None):
    try:
        # Fetch the document data from the API
        response = http.get(f"{api_url}/public_documents/{image_id}", auth=(username, password))
        response.raise_for_status()
        data = response.json()

        image_url = None
        
        # check if image has previously been processed
        if image_id in processed_ids:
            print(f"Image with ID: {image_id} has already been processed, skipping...")
            return False

        # Check the ID type
        if image_id.startswith("CCDB_"):
            # Check each field
            for field in ccdb_fields:
                # Get the image URL from the API response
                image_url = data.get(field)
                if image_url:
                    break

        elif image_id.startswith("CIL_"):
            # Remove the "CIL_" prefix
            id_number = image_id[4:]

            # Construct the image URL
            image_url = f"https://cildata.crbs.ucsd.edu/media/thumbnail_display/{id_number}/{id_number}_thumbnailx512.jpg"



        if image_url is not None:
            # Fetch the image data
            response = requests.get(image_url, stream=True, timeout=5)
            response.raise_for_status()

            # Load the image data with PIL
            image = Image.open(BytesIO(response.content))

            # Perform the initial cropping (to remove letterbox)
            image = crop_image(image)

            # Optional: Crop to target aspect ratio
            if crop_ratio:
                print(f"Cropping to target aspect ratio: {crop_ratio}")
                image = crop_to_aspect_ratio(image, crop_ratio)



            # Check if the processing is required
            if process:

                # Calculate the brightness, contrast, and entropy
                # BRIGHTNESS 0-1
                brightness = calculate_brightness(image)
                # CONTRAST 1-255
                contrast = calculate_contrast(image)
                # ENTROPY 1-8
                entropy = calculate_entropy(image)

                # Print the brightness, contrast, and entropy
                print(f"brightness: {brightness}, contrast: {contrast}, entropy: {entropy}")

                # hardcoded threshold to compare images against
                # brightness_min = 0.1
                # brightness_max = 0.9
                # contrast_min = 20
                # entropy_max = 7

                brightness_min = 0
                brightness_max = 1
                contrast_min = 0
                entropy_max = 10

                # Check the image against your thresholds
                if brightness < brightness_max and brightness > brightness_min and contrast > contrast_min and entropy < entropy_max:
                    # Try to process the image
                    processed_image = process_image(image)

                    # If the processed image is not None, save it
                    if processed_image is not None:
                        filename = os.path.join(output_folder, f"{image_id}_{processed_image.size[0]}x{processed_image.size[1]}.png")
                        processed_image.save(filename, "PNG")
                        print("image passed threshold, proceed.")
                        return True
                    else:
                        print("image did not pass resolution check, skipping...")
                        return False
            else:
                filename = os.path.join(output_folder, f"{image_id}_{image.size[0]}x{image.size[1]}.png")
                image.save(filename, "PNG")
                return True


    except requests.exceptions.Timeout:
        print(f"request timed out for image ID: {image_id}. please check network connection.")

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"image not found for ID: {image_id}")
        elif e.response.status_code == 429:
            print(f"rate limit exceeded for image ID: {image_id}. please wait before sending more requests.")
        else:
            print(f"HTTP error occurred for image ID: {image_id}. error details: {str(e)}")

    except Exception as e:
        print(f"an error occurred for image ID: {image_id}. error details: {str(e)}")

    return False

# alternatively, use a seed for pseudo-random ID shuffle
# random.seed(666)
# random.shuffle(ids)

# Fetch the list of public IDs
response = requests.get(f"{api_url}/public_ids?from=0&size=50000", auth=(username, password))
response.raise_for_status()

# Get the list of IDs
ids = [hit['_id'] for hit in response.json()['hits']['hits']]

# Filter the IDs to only include those up to 50000 to avoid placeholder images
ids = [id for id in ids if int(id[4:]) <= 50000]  # Assumes all IDs start with 'CIL_' which they seem to

# Randomly shuffle the list of IDs
# with new seed for random based on current time
random.seed(time.time())
random.shuffle(ids)

# Initialize counter for downloaded images
downloaded_images = 0

# Initialize an index for the IDs list
index = 0

# define a final output aspect ratio
# crop_ratio = 16/9 # widescreen
# crop_ratio = 2.35/1 # cinemascope
crop_ratio = 4/3 # u know


while downloaded_images < num_images and index < len(ids):
    # Call with process=True to process the image or process=False to just download
    if download_and_maybe_process_image(ids[index], process=True, crop_ratio=crop_ratio):
        downloaded_images += 1
        print(f"downloading {ids[index]} ({downloaded_images} of {min(num_images, len(ids))})")
    index += 1

print("done.")
# Record the end time
end_time = time.time()

# Calculate and print the total execution time
total_time_sec= end_time - start_time
total_time_min=total_time_sec/60
print(f"total runtime: {round(total_time_min, 2)} minutes")