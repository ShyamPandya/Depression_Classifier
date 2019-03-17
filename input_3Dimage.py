# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651
import csv
#import cv2
import os
import tensorflow as tf
from pip._vendor.progress.bar import Bar

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import nibabel as nib
import matplotlib.pyplot as plt
from dicom import logger
from medpy.filter import anisotropic_diffusion, gauss_xminus1d
from medpy.io import load, get_pixel_spacing
from medpy.filter import otsu
from nilearn import image as img


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("IMAGE_WIDTH",80,"image width")
tf.app.flags.DEFINE_integer("IMAGE_HEIGHT",80,"image heigth")
tf.app.flags.DEFINE_integer("IMAGE_DEPTH",50,"image depth")
batch_index = 0
filenames = []

# user selection
tf.app.flags.DEFINE_string("DATA_DIRECTORY","C:\\Users\SHYAM\Documents\Minor Project\Data","data directory")

tf.app.flags.DEFINE_integer("NUMBER_OF_CLASSES",2,"num of classes")

def get_filenames():
    with open(FLAGS.DATA_DIRECTORY +'\participants.tsv') as csvfile:
        reader=csv.DictReader(csvfile,delimiter='\t')
        for row in reader:
            filenames.append(row['participant_id'])

    print(filenames)


def get_data_jpeg(sess, data_set, batch_size):
    global batch_index, filenames

    get_filenames()
    max = len(filenames)
    print("max")
    print(max)
    batch_index = 0
    begin = batch_index
    end = batch_index + batch_size

    x_data = np.array([], np.float32)
    y_data = np.array([],np.float32) # zero-filled list for 'one hot encoding'
    print("Y_DATA SHAPEE")
    print(y_data.shape)

    #print(begin,end)
    for i in range(begin, end):
        print(filenames[i])
        imagePath = FLAGS.DATA_DIRECTORY + '\\' + 'ds171_R1.0.0' + '\\' + filenames[i] + '\\' + 'func' +'\\' + filenames[i] + '_task-nonmusic_run-4_bold.nii'
        print(imagePath)
        FA_org = nib.load(imagePath)
        FA_data, image_header = load(imagePath)
        FA_data = FA_org.get_data()  # 256x256x176; numpy.ndarray
        print(FA_data.shape)
        print(type(FA_data))


        #data_output = FA_data > threshold
        # apply the watershed
        logger.info('Applying anisotropic diffusion with settings: niter={} / kappa={} / gamma={}...'.format(100,0.2,0.4))
        #data_output = anisotropic_diffusion(FA_data,100,0,2,0.4,get_pixel_spacing(image_header))
        #data_output=gauss_xminus1d(FA_data,1,3)
        '''fig = plt.figure()
        for num, each_slice in enumerate(FA_data[:10]):
            y = fig.add_subplot(2,5,num+1)
            new_img = cv2.resize(np.array(each_slice),150)
            y.imshow(new_img)
        plt.show()'''



        # save file
        #np.save(data_output,'/output', image_header, 2)
        # TensorShape([Dimension(256), Dimension(256), Dimension(176)])
        resized_image = tf.image.resize_images(images=FA_data, size=(FLAGS.IMAGE_WIDTH,FLAGS.IMAGE_HEIGHT), method=1)
        print("x_data")
        print(x_data.shape)
        print("resized_image")
        print(resized_image.shape)
        image = sess.run(resized_image) # (256,256,40)
        print("image shape")
        print(image.shape)
        x_data = np.append(x_data, np.asarray(image, dtype='float32')) # (image.data, dtype='float32')
        y_data = np.append(y_data,0)

    batch_index += batch_size  # update index for the next batch
    print("x data shaope")
    print(x_data.shape)
    x_data_ = x_data.reshape(batch_size, FLAGS.IMAGE_WIDTH * FLAGS.IMAGE_HEIGHT * FLAGS.IMAGE_DEPTH)
    #y_data_ = y_data.reshape(batch_size, FLAGS.height * FLAGS.width * FLAGS.depth, -1)
    print("x data shaope")
    print(len(x_data))
    print("y data shape")
    print(y_data)

    return x_data_, y_data


def PCA_calculate(data):
    nsamples, nx, ny = data.shape
    X_std = StandardScaler().fit_transform(data.reshape((nsamples,nx*ny)))
    mean_vec = np.mean(X_std, axis=0)
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
    print('Covariance matrix \n%s' % cov_mat)

    #print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)

    cor_mat1 = np.corrcoef(X_std.T)

    eig_vals, eig_vecs = np.linalg.eig(cor_mat1)

    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)

    cor_mat2 = np.corrcoef(data.T)

    eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)
    u, s, v = np.linalg.svd(X_std.T)
    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Everything ok!')

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort()
    eig_pairs.reverse()

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i[0])

    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    trace1 = Bar(
        x=['PC %s' % i for i in range(1, 5)],
        y=var_exp,
        showlegend=False)

    trace2 = Scatter(
        x=['PC %s' % i for i in range(1, 5)],
        y=cum_var_exp,
        name='cumulative explained variance')

    data = Data([trace1, trace2])