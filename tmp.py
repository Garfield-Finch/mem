import numpy as np
from PIL import Image
import matplotlib

file_name = '/p300/dataset/iPER/kpt_imgs/001/1/1/000.npy'

img_np = np.load(file_name)

img_np = np.sum(img_np, axis=2)
print(img_np.dtype)
print(img_np.shape)

# matplotlib.image.imsave('name.png', img_np0)

img_np = img_np * 255
im = Image.fromarray(img_np)
if im.mode == "F":
    im = im.convert('RGB')
im.save("your_file.png")