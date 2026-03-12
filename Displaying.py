# affichage / sauvegarde d'images chips PlanesNet

import json
import numpy as np
from matplotlib import pyplot as plt

with open(r"Data/planesnet/planesnet.json") as f:
    planesnet = json.load(f)

print(planesnet.keys())
print("label ligne 300:", planesnet["labels"][300])

index = 300
im = np.array(planesnet["data"][index]).astype("uint8")
im = im.reshape((3, 400)).T.reshape((20, 20, 3))
print("shape:", im.shape)

plt.imshow(im)
plt.show()

# pour sauvegarder: Image.fromarray(im).save('Test_img/test.png')
