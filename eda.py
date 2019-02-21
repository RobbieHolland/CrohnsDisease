'''
Generates TFRecords of the dataset
'''

import random
import os
import cv2
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from dltk.io.augmentation import *
from dltk.io.preprocessing import *

from data.handling.parse_labels import DataParser
from scripts.show import *

def contours(image):
    # load image
    gray = np.array(image).astype(np.uint8)

    # gray = cv2.cvtColor(image, cv2.COLOR_GRAY2GRAY) # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, \
                                       type=cv2.THRESH_BINARY_INV)

    image, contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, \
                                       cv2.CHAIN_APPROX_SIMPLE)

    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    print('-------')
    cnt = contours[10]
    img = cv2.drawContours(image, [cnt], 0, 255, 3)
    cv2.imshow("window title", img)
    cv2.waitKey()

    print('theyre drawn')

    # for cont in contours:
    #     x,y,w,h = cv2.boundingRect(cont)
    #     c_x, c_y = x + w/2, y + h/2
    #     print(c_x, c_y, w, h)
    #     area = w*h
    #     if c_x < 200 and c_x > 56 and c_y < 200 and c_y > 56 and w < 256 and h < 256 and w > 80 and h > 80 and area > mx_area:
    #         print('yes!')
    #         mx = x,y,w,h
    #         mx_area = area
    # x,y,w,h = mx
    # print(x + w/2, y + h/2, w, h)
    #
    # # Output to files
    # roi=image[y:y+h,x:x+w]

    # cv2.imshow('img', roi)

def blobs(image):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = -500
    params.maxThreshold = 500

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 5000
    params.maxArea = 65000

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.01

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.005

    # Filter by Inertia
    params.filterByColor = False
    params.filterByInertia = False
    params.minInertiaRatio = 0.01


    image = np.array(image).astype(np.uint8)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(image)
    # print('Keypoints', [k.pt for k in keypoints])
    # print('Size', [k.size for k in keypoints])
    print(len(keypoints))

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS # ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # #
    # # # Show keypoints
    # cv2.imshow("Keypoints", im_with_keypoints)
    # cv2.waitKey(0)

def resize(image):
    itk_image = sitk.GetImageFromArray(np.array(image))
    resample = sitk.ResampleImageFilter()
    scale = sitk.ScaleTransform(2, (2, 2))
    resample.SetTransform(scale)
    resample.SetSize((256, 256))
    itk_image = resample.Execute(itk_image)
    scaled_image = sitk.GetArrayFromImage(itk_image)

    return scaled_image

def pre_process(image):
    # Normalise image
    image = whitening(image)
    return image

def register2(fixed, image):
    fixed = sitk.GetImageFromArray(fixed)

    moving = sitk.GetImageFromArray(image)
    initial_transform = sitk.CenteredTransformInitializer(fixed,
                                                      moving,
                                                      sitk.Euler2DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    moving_resampled = sitk.Resample(moving, fixed, initial_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())
    arr = sitk.GetArrayFromImage(moving_resampled)
    print(arr)
    return arr


def register(fixed, image):
    x = image.astype(np.int64)

    fixed = sitk.GetImageFromArray(fixed)

    moving = sitk.GetImageFromArray(image)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200)
    R.SetInitialTransform(sitk.AffineTransform(fixed.GetDimension()))
    R.SetInterpolator(sitk.sitkLinear)

    R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )
    outTx = R.Execute(fixed, moving)
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))
    # sitk.WriteTransform(outTx,  sys.argv[3])


    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)


    out = resampler.Execute(moving)
    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)

    arr = sitk.GetArrayFromImage(out)
    print(arr.shape)
    return arr


def plot_array(array):
    itk_image = sitk.GetImageFromArray(np.array(array))
    # sitk.Show(itk_image)
    print(array.shape)
    imgplot = plt.imshow(itk_image)
    plt.show()

def volume_index(record, slice):
    return record.volume_height - 1 - slice

class ImageEDA:
    def __init__(self, label_path, data_path, out_path):
        self.reader = DataParser(label_path, data_path)
        self.data_path = data_path
        self.metadata = self.reader.shuffle_read()

        self.metadata.print_statistics()

    def show(self):
        fixed_image = None
        for record in self.metadata.records:
            if len(record.slices) == 0:
                continue

            print('Loading', record.patient_no)
            data = sitk.ReadImage(record.form_path(self.data_path))
            volume = sitk.GetArrayFromImage(data)
            print(record.patient_position)
            print(volume.shape)
            label = record.polyp_class
            print('with label', label)


            for slice in record.slices:
                index = volume_index(record, slice)

                image = np.array(volume[index])
                if fixed_image is None:
                    fixed_image = image

                image = resize(image)
                image = pre_process(image)
                image = register(fixed_image, image)
                # blobs(image)

                show(image)

eda = ImageEDA('./data/cases/', '/vol/bitbucket/rh2515/CT_Colonography', 'data/tfrecords')
eda.show()
