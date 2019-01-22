import os
import pydicom

def header():
    return ['patient_no', 'path', 'volume_height', 'position', 'slice_no', 'file_name', 'polyp_class']

class StudyRecord:
    def __init__(self, patient_no, path, slices, volume_height, patient_position, class_number):
        self.patient_no = patient_no
        self.study_path = path
        self.slices = slices
        self.slice_files = []
        self.volume_height = volume_height
        self.patient_position = patient_position
        self.polyp_class = class_number

        self.is_abnormal = self.polyp_class > 0

    def set_slice_files(self, slice_files):
        self.slice_files = slice_files

    def csv_rows(self):
        rows = []
        patient_no = self.study_path.split('/')[1]
        for i in range(len(self.slice_files)):
            rows.append([patient_no, self.study_path, self.volume_height, self.patient_position, self.slices[i], self.slice_files[i], self.polyp_class])
        return rows
