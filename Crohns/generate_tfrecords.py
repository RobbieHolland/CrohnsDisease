import SimpleITK as sitk
import numpy as np

from preprocess import Preprocessor
from metadata import Metadata
from tfrecords import TFRecordGenerator

reference_size = [256, 128, 64]
test_proportion = 0.2
data_path = '/vol/bitbucket/rh2515/MRI_Crohns'
record_out_path = '/vol/bitbucket/rh2515/Crohns/tfrecords'
record_suffix = 'axial_t2_only'
metadata = Metadata('/vol/bitbucket/rh2515/MRI_Crohns')

for patient in metadata.patients:
    patient.set_images(sitk.ReadImage(patient.axial))

preprocessor = Preprocessor(constant_volume_size=reference_size)

metadata.patients = preprocessor.process(metadata.patients)

record_generator = TFRecordGenerator(record_out_path, record_suffix)

features = list([patient.axial_image for patient in metadata.patients])
labels = [patient.get_label() for patient in metadata.patients]
record_generator.generate_train_test(test_proportion, features, labels)

print('Done')
