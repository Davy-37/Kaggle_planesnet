import json
import numpy as np
from PIL import Image

# ouverture du json planesnet
with open("Data/planesnet/planesnet.json", "r") as f:
    dataset = json.load(f)

data = np.array(dataset["data"])
labels = np.array(dataset["labels"])
scene_ids = dataset["scene_ids"]
locations = dataset["locations"]

n_images = len(labels)
print("Nombre d'images :", n_images)

# recréer la 1ère image (R, G, B en 20x20)
img_data = np.array(data[0])
R = img_data[:400].reshape((20, 20))
G = img_data[400:800].reshape((20, 20))
B = img_data[800:].reshape((20, 20))

img = np.dstack((R, G, B)).astype(np.uint8)
Image.fromarray(img).show()
