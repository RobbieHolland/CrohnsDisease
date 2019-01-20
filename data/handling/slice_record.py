import os
import pydicom

def slice_record_from_row(r):
    return SliceRecord(r[0], r[1], r[3], r[4], r[5], r[6])

class SliceRecord:
    def __init__(self, patient_no, path, patient_position, slice_no, file_name, class_number):
        self.patient_no = patient_no
        self.study_path = path
        self.slice_no = slice_no
        self.file_name = file_name
        self.patient_position = patient_position
        self.polyp_class = int(class_number)

        self.is_abnormal = self.polyp_class > 0

    def slice_path(self, data_path):
        return os.path.join(data_path, self.study_path, str(self.file_name).zfill(6) + '.dcm')

    def load_slice_image(self, data_path):
        slice_data = pydicom.dcmread(self.slice_path(data_path))
        return slice_data.pixel_array
