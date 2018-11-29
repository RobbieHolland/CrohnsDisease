import SimpleITK as sitk
import os
import tensorflow as tf
import pandas as pd
import time
import matplotlib.pyplot as plt

from dltk.io.augmentation import *
from dltk.io.preprocessing import *

data_path = 'copied_data/clean_nifti/'
data = os.listdir(data_path)

for d in data:
    t1_fn = os.path.join(data_path, d)

    # Read image
    f"(Reading data ${data_path})"
    sitk_t1 = sitk.ReadImage(t1_fn)
    t1 = sitk.GetArrayFromImage(sitk_t1)

    # Normalise the image to zero mean/unit std dev:
    t1 = whitening(t1)

    # Other
    print(t1.shape)
    imgplot = plt.imshow(t1[300])

    plt.show()
