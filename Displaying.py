#Displaying and saving image chips


import json
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

#with open("Data/planesnet/planesnet.json", "r") as file:
#    loaded_data = json.load(file)
#    print(loaded_data)


f = open(r'Data/planesnet/planesnet.json')
planesnet = json.load(f)
f.close()
print(planesnet.keys())
print(planesnet['labels'][300])

index = 300 # Row to be saved
im = np.array(planesnet['data'][index]).astype('uint8')
im = im.reshape((3, 400)).T.reshape((20,20,3))
print(im.shape)

plt.imshow(im)
plt.show()


#out_im = Image.fromarray(im)
#out_im.save('Test_img/test.png')
