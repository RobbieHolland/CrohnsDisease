import pydicom
import os

class SliceToFile:
    def __init__(self, data_path):
        self.data_path = data_path

    def convert(self, record, candidate_slices):
        mappings = []
        for file_no in range(int(record.volume_height)):
            file = os.path.join(self.data_path, record.study_path, str(file_no).zfill(6) + '.dcm')
            ds = pydicom.dcmread(file, stop_before_pixels=True)
            mappings.append([ds.InstanceNumber, file_no])

        slices = []
        file_names = []
        for slice in candidate_slices:
            matching_files = [m[1] for m in mappings if m[0] == slice]
            if len(matching_files) == 1:
                file_names.append(matching_files[0])
                slices.append(slice)
        return slices, file_names
