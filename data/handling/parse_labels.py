import csv
import os
import random
import pydicom
from data.handling.study_record import StudyRecord
from data.handling.metadata import Metadata
random.seed(1234)

# Slices are of the form 'Pos >=10mm 34/45/63/109'
def parse_slices(slices):
    return [int(slice) for slice in
            slices.replace('Pos >=10mm', '').replace(' ', '').split('/') if slice.isdigit()]

def parse_abnormal_row(row):
    # [Case ID, [supine slice numbers], [prone slice numbers]]
    return [row[0], parse_slices(row[1]), parse_slices(row[2])]

def parse_path_position(row):
    # Patient no, Path, Volume height, Patient position
    return (row[0].split('/')[1], row[0], row[1], row[2])

def parse_slice_to_file(row):
    # Study path, slice number, file number
    return (row[0], int(row[1]), int(row[2]))

def read_csv(path, file, row_parser, header=True):
    parsed_data = []
    with open(os.path.join(path, file + '.csv'), 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data = row_parser(row)
            parsed_data.append(data)
        # Discard column headers
        if header:
            return parsed_data[1:]
        return parsed_data

class DataParser:
    def __init__(self, label_path, data_path):
        self.label_path = label_path
        self.data_path = data_path
        self.relevant_positions = [['FFS', 'HFS'], ['FFP', 'HFP']]
        self.n_slices_per_healthy = 1
        self.neighbour_distance = 2

    def _record_is_series(self, record):
        n_mbytes = os.path.getsize(record.form_path(self.data_path)) >> 20
        return n_mbytes > 100 and record.volume_height > 100

    def parse_records(self, class_number, polyp_records, studies):
        records = []
        for record in polyp_records:
            patient_no = record[0]
            matching_studies = [s for s in studies if s[0] == patient_no]

            relevant_records = []
            for i, position in enumerate(self.relevant_positions):
                polyp_slices = record[i + 1]

                next_records = [StudyRecord(patient_no, polyp_slices, int(s[2]), s[3], class_number)
                                   for s in matching_studies if s[3] in position]

                # If there are two matching studies, use neither
                if len(next_records) == 1:
                    relevant_records += next_records
            records += relevant_records
        return records

    def shuffle_read(self):
        metadata = Metadata()

        # Read metadata
        studies = read_csv(self.label_path, 'studies_positions', parse_path_position, header=False)
        ten_plus = read_csv(self.label_path, '10+', parse_abnormal_row)
        six_to_nine = read_csv(self.label_path, '6-9', parse_abnormal_row)
        no_polyp = read_csv(self.label_path, 'None', lambda x: [x[0], [], []])

        # Abnormal studies
        polyps = [[1, six_to_nine], [2, ten_plus]]
        for class_number, polyp_class in polyps:
            metadata.add_records(self.parse_records(class_number, polyp_class, studies))
        bad_positive_records = read_csv(self.label_path, 'bad_positive_conversions', lambda x: x[0])
        metadata.filter_records(lambda r: r.patient_no not in bad_positive_records)

        # Healthy studies
        no_polyp = random.sample(no_polyp, metadata.n_abnormal_patients())
        metadata.add_records(self.parse_records(0, no_polyp, studies))

        # Filter records for series
        metadata.filter_records(lambda r: self._record_is_series(r))

        # Generate all slices from centroids
        metadata.compute_slices(self.n_slices_per_healthy, self.neighbour_distance)

        random.shuffle(metadata.records)
        # metadata.cut_dataset(40)
        return metadata
