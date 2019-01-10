import os

class StudyRecord:
    def __init__(self, patient_no, path, tumour_slices, volume_height, patient_position):
        self.patient_no = patient_no
        self.path = path
        self.tumour_slices = tumour_slices
        self.volume_height = volume_height
        self.patient_position = patient_position

    def slice_path(self, slice):
        return os.path.join(self.path, str(slice).zfill(6) + '.dcm')

    def __str__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())
        # return str(self.volume_height)
