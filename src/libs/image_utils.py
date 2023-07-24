import numpy as np
import cv2
from .register import IMAGE_TRANSFORM

@IMAGE_TRANSFORM.registry()
def random_offset(img, p=0.1):

    h, w = img.shape[:2]
    oh = np.random.randint(-int(h * p), int(h * p))
    ow = np.random.randint(-int(w * p), int(w * p))
    M = np.asarray([[1, 0, oh], 
                    [0, 1, ow]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (h, w))

    return img

@IMAGE_TRANSFORM.registry()
def random_rotate(img, p=0.05):

    r = np.random.randint(-int(360 * p), int(360 * p))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((h / 2, w / 2), angle=r, scale=1)
    img = cv2.warpAffine(img, M, (h, w))

    return img

@IMAGE_TRANSFORM.registry()
def random_flip(img, p=0.5):

    if np.random.random() < p:
        img = img[::-1]
    if np.random.random() < p:
        img = img[:, ::-1]

    return img

@IMAGE_TRANSFORM.registry()
def random_crop(img, type='size', scale=(0.5, 1), size=(224, 224), center=False, keep=False):

    h, w = img.shape[:2]

    if type  == 'size':
        new_h, new_w = size
    else:
        ratio = np.random.uniform(*scale)
        new_h, new_w = int(ratio * h), int(ratio * w)

    if center:
        top = h // 2 - new_h // 2
        left = w //2 - new_w // 2
    else:
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

    img = img[top:top+new_h, left:left+new_w]

    if keep:
        img = cv2.resize(img, (h, w))

    return img
