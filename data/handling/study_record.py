import os
import pydicom

def header():
    return ['patient_no', 'volume_height', 'position', 'slice_no', 'polyp_class']

class StudyRecord:
    def __init__(self, patient_no, slice_centroids, volume_height, patient_position, class_number, neighbour_distance):
        self.patient_no = patient_no
        self.slice_centroids = slice_centroids
        self.volume_height = volume_height
        self.patient_position = patient_position
        self.polyp_class = class_number
        self.neighbour_distance = neighbour_distance

        self.is_abnormal = self.polyp_class > 0

    def centroid_slice_proportion(self):
        return [centroid / self.volume_height for centroid in self.slice_centroids]

    def is_series(self, data_path):
        n_mbytes = os.path.getsize(self.form_path(data_path)) >> 20
        return n_mbytes > 100 and self.volume_height > 100

    def csv_rows(self):
        rows = []
        for i in range(len(self.n_centroids())):
            rows.append([self.patient_no, self.volume_height, self.patient_position, self.slice_centroids[i], self.polyp_class])
        return rows

    def form_path(self, data_path):
        return os.path.join(data_path, '{}/{}.nii'.format(self.patient_no, self.patient_position))

    def n_slice_centroids(self):
        return len(self.slice_centroids)

    def n_centroids(self):
        return len(self.slice_centroids)

    def __repr__(self):
        return self.patient_no
