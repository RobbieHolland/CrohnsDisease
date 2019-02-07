import csv
import os
import random
import pydicom
from data.handling.study_record import StudyRecord

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

# For healthy slices, first allocates random locations
# Then for all slices include neighbouring slices
def set_slices(records, nei_len):
    for record in records:
        all_slices = []
        n_healthy_slices = 2
        if not record.is_abnormal:
            record.slices = random.sample(range(100, record.volume_height - 100), n_healthy_slices)

        for slice in record.slices:
            all_slices += list(set([nei for nei in range(slice - nei_len, slice + nei_len + 1) if nei >= 0 and nei < record.volume_height - 1]))

        record.slices = all_slices
    return records

class DataParser:
    def __init__(self, label_path, data_path):
        self.label_path = label_path
        self.data_path = data_path
        self.relevant_positions = [['FFS', 'HFS'], ['FFP', 'HFP']]

    def _record_is_series(self, record):
        n_mbytes = os.path.getsize(record.form_path(self.data_path)) >> 20
        return n_mbytes > 100 and record.volume_height > 100

    def shuffle_read(self):
        records = []
        studies = read_csv(self.label_path, 'studies_positions', parse_path_position, header=False)

        ten_plus = read_csv(self.label_path, '10+', parse_abnormal_row)
        six_to_nine = read_csv(self.label_path, '6-9', parse_abnormal_row)
        n_abnormal = len(ten_plus) + len(six_to_nine)
        no_polyp = read_csv(self.label_path, 'None', lambda x: [x[0], [], []])
        random.shuffle(no_polyp)
        no_polyp_ixs = random.sample(range(len(no_polyp)), n_abnormal)
        no_polyp = [no_polyp[i] for i in no_polyp_ixs]

        polyps = [no_polyp, six_to_nine, ten_plus]
        for class_number, polyp_class in enumerate(polyps):
            for record in polyp_class:
                patient_no = record[0]
                matching_studies = [s for s in studies if s[0] == patient_no]

                relevant_records = []
                for i, position in enumerate(self.relevant_positions):
                    polyp_slices = record[i + 1]

                    next_records = [StudyRecord(patient_no, polyp_slices, int(s[2]), s[3], class_number)
                                       for s in matching_studies if s[3] in position]

                    # If there are two matching studies, use neither
                    if len(next_records) == 1:
                        next_records = set_slices(next_records, 2)
                        relevant_records += next_records
                records += relevant_records

        # Filter records for series
        series_records = [r for r in records if self._record_is_series(r)]

        random.shuffle(series_records)
        return series_records[1:40]
