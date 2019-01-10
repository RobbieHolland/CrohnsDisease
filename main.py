import pydicom
import matplotlib.pyplot as plt
from data.parse_labels import DataParser

reader = DataParser('./data/cases/', '/vol/bitbucket/bkainz/TCIA/CT COLONOGRAPHY')
labels = reader.read()

# Total number of tumours
print(sum([len(x.tumour_slices) for x in labels]))

# Load example image
ds = pydicom.dcmread(labels[0].slice_path(300))
print(ds.pixel_array.sum())
imgplot = plt.imshow(ds.pixel_array)
plt.show(imgplot)
