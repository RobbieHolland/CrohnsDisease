import os
import pydicom

class StudyRecord:
    def __init__(self, patient_no, path, polyp_slices, volume_height, patient_position):
        self.patient_no = patient_no
        self.path = path
        self.polyp_slices = polyp_slices
        self.volume_height = volume_height
        self.patient_position = patient_position

        self.is_abnormal = len(self.polyp_slices) > 0

    def slice_path(self, slice):
        return os.path.join(self.path, str(slice).zfill(6) + '.dcm')

    def load_slice_image(self, slice):
        slice_data = pydicom.dcmread(self.slice_path(slice))
        return slice_data.pixel_array

    def __str__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())
        # return str(self.volume_height)
