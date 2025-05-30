import requests
import config
from PIL import Image, ImageOps
from io import BytesIO
import os
import random
import time
import argparse

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

# Function to adjust image to cinema aspect ratio (2.39:1)
def adjust_to_cinema_aspect(image, cinema_aspect_ratio=2.39, output_size=(1920, 804)):
    """
    Adjusts the given image to a cinema aspect ratio (2.39:1) by cropping or padding as necessary.
    """
    width, height = image.size
    img_aspect_ratio = width / height

    if img_aspect_ratio > cinema_aspect_ratio:
        # Image is wider than the target aspect ratio, crop horizontally
        new_width = int(cinema_aspect_ratio * height)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    elif img_aspect_ratio < cinema_aspect_ratio:
        # Image is taller than the target aspect ratio, pad vertically
        new_height = int(width / cinema_aspect_ratio)
        image = ImageOps.pad(image, (width, new_height), color=(0, 0, 0))

    # Resize the image to the output size (1920x804 for cinema aspect ratio)
    image = image.resize(output_size, Image.LANCZOS)
    return image

# Function to get a random image from the API and adjust to cinema aspect ratio
def download_image(image_id, output_folder):
    try:
        # Fetch the document data from the API
        response = requests.get(f"{api_url}/public_documents/{image_id}", auth=(username, password), timeout=5)
        response.raise_for_status()
        data = response.json()

        # Check the ID type
        if image_id.startswith("CCDB_"):
            # Check each field
            for field in ccdb_fields:
                # Get the image URL from the API response
                image_url = data.get(field)
                if image_url:
                    # Fetch the image data
                    response = requests.get(image_url, stream=True, timeout=5)
                    response.raise_for_status()

                    # Load the image data with PIL
                    image = Image.open(BytesIO(response.content))

                    # Adjust the image to cinema aspect ratio and resize
                    image = adjust_to_cinema_aspect(image)

                    # Save the image
                    filename = os.path.join(output_folder, f"{image_id}_cinema.jpg")
                    image.save(filename, "JPEG")
                    print(f"Image {image_id} saved as {filename}")
        elif image_id.startswith("CIL_"):
            # Remove the "CIL_" prefix
            id_number = image_id[4:]

            # Construct the image URL
            image_url = f"https://cildata.crbs.ucsd.edu/media/thumbnail_display/{id_number}/{id_number}_thumbnailx512.jpg"

            # Fetch the image data
            response = requests.get(image_url, stream=True, timeout=5)
            response.raise_for_status()

            # Load the image data with PIL
            image = Image.open(BytesIO(response.content))

            # Adjust the image to cinema aspect ratio and resize
            image = adjust_to_cinema_aspect(image)

            # Save the image
            filename = os.path.join(output_folder, f"{image_id}_cinema.jpg")
            image.save(filename, "JPEG")
            print(f"Image {image_id} saved as {filename}")

    except requests.exceptions.Timeout:
        print(f"Request timed out for image ID: {image_id}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"Image not found for ID: {image_id}")
        else:
            print(f"HTTP error occurred for image ID: {image_id}. Error details: {str(e)}")
    except Exception as e:
        print(f"An error occurred for image ID: {image_id}. Error details: {str(e)}")

def main(num_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Fetch the list of public IDs
    response = requests.get(f"{api_url}/public_ids?from=0&size=50000", auth=(username, password))
    response.raise_for_status()

    # Get the list of IDs
    ids = [hit['_id'] for hit in response.json()['hits']['hits']]

    # Randomly shuffle the list of IDs
    random.seed(time.time())
    random.shuffle(ids)

    # Download the images
    for i in range(min(num_images, len(ids))):
        download_image(ids[i], output_folder)
        print(f"Downloading... ({i+1} of {min(num_images, len(ids))})")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from the CIL API and adjust to cinema aspect ratio")
    parser.add_argument("num_images", type=int, help="Number of images to download")
    parser.add_argument("output_folder", help="Path to the output folder for downloaded images")

    args = parser.parse_args()

    num_images = args.num_images
    output_folder = args.output_folder

    main(num_images, output_folder)
