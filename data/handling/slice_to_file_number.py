import pydicom
import os

class SliceToFile:
    def __init__(self, data_path):
        self.data_path = data_path

    def convert(self, record):
        mappings = []
        for file_no in range(int(record.volume_height)):
            file = os.path.join(self.data_path, record.study_path, str(file_no).zfill(6) + '.dcm')
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            mappings.append([ds.InstanceNumber, file_no])

        file_names = []
        for slice in record.slices:
            file_name = [m[1] for m in mappings if m[0] == slice][0]
            file_names.append(file_name)

        return file_names
