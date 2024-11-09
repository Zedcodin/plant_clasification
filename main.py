import os
import numpy as np
from PIL import Image
from datasets import load_dataset
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# Assuming 'images' directory exists and contains images
IMAGE_FOLDER = "images"
image_uris = sorted([os.path.join(IMAGE_FOLDER, image_name) for image_name in os.listdir(IMAGE_FOLDER)])
ids = [str(i) for i in range(len(image_uris))]

client = chromadb.Client()
embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()
collection = client.create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=image_loader
)

print("hi")
single_image_uri = image_uris[0]
single_id = ids[0]

try:
    collection.add(ids=[single_id], uris=[single_image_uri])
    print("Successfully added a single image.")
except Exception as e:
    print(f"An error occurred during single image add operation: {e}")

print("hiiis")

query_image_path = "./images/jb1.jpg"  # Adjust path as needed
query_image = np.array(Image.open(query_image_path))
retrieved = collection.query(query_images=[query_image], include=['data'], n_results=3)

print(retrieved)
