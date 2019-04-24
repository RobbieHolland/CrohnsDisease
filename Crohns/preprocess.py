import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import math

def show_data(data, sl):
    fig = plt.figure(figsize=(18, 18))
    fig.set_size_inches(15, 10)
    columns = 8
    rows = math.ceil(len(data) / columns)
    for i, image in enumerate(data):
        fig.add_subplot(rows, columns, i + 1)
        nda = sitk.GetArrayFromImage(image) / 255
        nda = nda.astype(np.float32)

        plt.imshow(nda[sl], cmap='gray')
    plt.show()

class Preprocessor:
    def generate_reference_volume(self, patients):
        # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
        reference_physical_size = np.zeros(self.dimension)
        for patient in patients:
            img = patient.axial_image
            reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

        # Create the reference image with a zero origin, identity direction cosine matrix and dimension
        reference_origin = np.zeros(self.dimension)
        reference_direction = np.identity(self.dimension).flatten()
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(self.constant_volume_size, reference_physical_size) ]

        reference_image = sitk.Image(self.constant_volume_size, patients[0].axial_image.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
        # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
        # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
        # spacing will not yield the correct coordinates resulting in a long debugging session.
        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

        return reference_image

    def __init__(self, constant_volume_size=[256, 128, 64]):
        self.constant_volume_size = constant_volume_size

    def process(self, patients):
        print('Preprocessing...')
        self.dimension = patients[0].axial_image.GetDimension()
        reference_volume = self.generate_reference_volume(patients)
        reference_center = np.array(reference_volume.TransformContinuousIndexToPhysicalPoint(np.array(reference_volume.GetSize())/2.0))

        # Crop
        print('Cropping volumes')
        for patient in patients:
            print(f'Cropping {patient.get_id()} \r', end='')
            patient.set_images(axial_image=self.threshold_based_crop(patient.axial_image))
        show_data([p.axial_image for p in patients], 30)

        # Resample
        print(f'Resampling volumes to {self.constant_volume_size}')
        for patient in patients:
            patient.set_images(axial_image=self.resample(patient, reference_volume, reference_center))
        show_data([p.axial_image for p in patients], 30)

        return patients

    def resample(self, patient, reference_volume, reference_center):
        img = patient.axial_image

        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.TranslationTransform(self.dimension)
        transform.SetOffset(np.array(img.GetOrigin()) - reference_volume.GetOrigin())
        # transform.SetMatrix(img.GetDirection())
        centered_transform = sitk.Transform(transform)

        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(self.dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform.AddTransform(centering_transform)

        # Using the linear interpolator as these are intensity images
        return sitk.Resample(img, reference_volume, centered_transform, sitk.sitkLinear, 0.0)

    def threshold_based_crop(self, image):
        inside_value = 20
        outside_value = 255
        label = 1
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        seed = (int(image.GetSize()[0]/2), int(image.GetSize()[1]/2), int(image.GetSize()[2]/2))
        perturbations = [-5, 5]
        seeds = [seed]
        seeds += [(seed[0], seed[1], seed[2] + p) for p in perturbations]
        seeds += [(seed[0], seed[1] + p, seed[2]) for p in perturbations]

        seg_explicit_thresholds = sitk.ConnectedThreshold(image, seedList=seeds,
                                                          lower=inside_value, upper=outside_value)
        overlay = sitk.LabelOverlay(image, seg_explicit_thresholds)
        label_shape_filter.Execute( seg_explicit_thresholds )
        bounding_box = label_shape_filter.GetBoundingBox(label)

        return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
