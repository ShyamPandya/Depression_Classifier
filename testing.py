import tensorflow as tf
import csv
import numpy as np
from nilearn import image
import math
import nibabel as nib
from random import shuffle
from nilearn.plotting import plot_epi
from nilearn import plotting


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("IMAGE_WIDTH",80,"image width")
tf.app.flags.DEFINE_integer("IMAGE_HEIGHT",80,"image heigth")
tf.app.flags.DEFINE_integer("IMAGE_DEPTH",50,"image depth")
filenames =[]




# user selection
tf.app.flags.DEFINE_string("DATA_DIRECTORY","C:\\Users\SHYAM\Documents\Minor Project\Data","data directory")

tf.app.flags.DEFINE_integer("NUMBER_OF_CLASSES",3,"num of classes")



def get_data(batch_size,filenames):
    shuffle(filenames)
    tones = np.array([], np.float32)
    negative_music = np.array([], np.float32)
    positive_music = np.array([], np.float32)
    y_data_negative = np.array([], np.float32)
    y_data_positive = np.array([], np.float32)
    for i in range(0,batch_size):
        print(filenames[i])
        for j in range(1,4):
            print(j)
            imageMusicPath = FLAGS.DATA_DIRECTORY + '\ds171_R1.0.0' + '\\' + filenames[i] + '\\' + 'func' + '\\' + filenames[i] + '_task-music_run-' + str(j) + '_bold.nii'
            detailMusicPath = FLAGS.DATA_DIRECTORY + '\ds171_R1.0.0' + '\\' + filenames[i] + '\\' + 'func' + '\\' + filenames[i] + '_task-music_run-' + str(j) + '_events.tsv'
            with open(detailMusicPath) as csvfile:
                reader=csv.DictReader(csvfile, delimiter='\t')
                onset = 0
                flag =''
                base = 105/312
                for row in reader:
                    if row['trial_type'] == 'tones':
                        onset = onset + float(row['duration'])
                        flag = 't'
                    elif row['trial_type'] == 'negative_music':
                        onset = onset + float(row['duration'])
                        flag = 'n'
                    elif row['trial_type'] == 'positive_music':
                        onset = onset + float(row['duration'])
                        flag = 'p'
                    elif row['trial_type'] == 'response':
                        temp = nib.load(imageMusicPath)
                        print(temp.shape)
                        tempLoad = image.index_img(imageMusicPath,math.floor((onset+(float(row['duration'])/2))*base))
                        tempData = tempLoad.get_data()
                        print(tempData.shape)
                        resizedImage = tf.image.resize_images(images=tempData, size=[FLAGS.IMAGE_WIDTH,FLAGS.IMAGE_HEIGHT], method=1)
                        print(type(resizedImage))
                        onset = onset + int(row['duration'])
                        if flag == 't':
                            tones = np.append(tones, np.asarray(tempData, dtype='float32'))
                            #if 'mdd' in filenames[i]:
                                #y_data = np.append(y_data, np.asarray(1, dtype='float32'))
                            #else:
                                #y_data = np.append(y_data, np.asarray(0, dtype='float32'))
                        if flag == 'n':
                            negative_music = np.append(negative_music, np.asarray(tempData, dtype='float32'))
                            if 'mdd' in filenames[i]:
                                y_data_negative = np.append(y_data_negative, np.asarray(1, dtype='float32'))
                            else:
                                y_data_negative = np.append(y_data_negative, np.asarray(0, dtype='float32'))
                        if flag == 'p':
                            positive_music = np.append(positive_music, np.asarray(tempData, dtype='float32'))
                            if 'mdd' in filenames[i]:
                                y_data_positive = np.append(y_data_positive, np.asarray(1, dtype='float32'))
                            else:
                                y_data_positive = np.append(y_data_positive, np.asarray(0, dtype='float32'))
                        #print(negative_music.shape)
        for k in range(4,6):
            print(k)
            imageNonMusicPath = FLAGS.DATA_DIRECTORY + '\ds171_R1.0.0' + '\\' + filenames[i] + '\\' + 'func' + '\\' + filenames[i] + '_task-nonmusic_run-' + str(k) + '_bold.nii'
            detailNonMusicPath = FLAGS.DATA_DIRECTORY + '\ds171_R1.0.0' + '\\' + filenames[i] + '\\' + 'func' + '\\' + filenames[i] + '_task-nonmusic_run-' + str(k) + '_events.tsv'
            with open(detailNonMusicPath) as csvfile:
                reader=csv.DictReader(csvfile, delimiter='\t')
                onset = 0
                flag =''
                base = 105/312
                for row in reader:
                    if row['trial_type'] == 'tones':
                        onset = onset + float(row['duration'])
                        flag = 't'
                    elif row['trial_type'] == 'negative_nonmusic':
                        onset = onset + float(row['duration'])
                        flag = 'n'
                    elif row['trial_type'] == 'positive_nonmusic':
                        onset = onset + float(row['duration'])
                        flag = 'p'
                    elif row['trial_type'] == 'response':
                        temp = nib.load(imageNonMusicPath)
                        print(temp.shape)
                        tempLoad = image.index_img(imageNonMusicPath,math.floor((onset+(float(row['duration'])/2))*base))
                        tempData = tempLoad.get_data()
                        print(tempData.shape)
                        resizedImage = tf.image.resize_images(images=tempData, size=[FLAGS.IMAGE_WIDTH,FLAGS.IMAGE_HEIGHT], method=1)
                        print(type(resizedImage))
                        onset = onset + int(row['duration'])
                        if flag == 't':
                            tones = np.append(tones, np.asarray(tempData, dtype='float32'))
                            #if 'mdd' in filenames[i]:
                             #   y_data = np.append(y_data, np.asarray(1, dtype='float32'))
                            #else:
                             #   y_data = np.append(y_data, np.asarray(0, dtype='float32'))
                        if flag == 'n':
                            negative_music = np.append(negative_music, np.asarray(tempData, dtype='float32'))
                            if 'mdd' in filenames[i]:
                                y_data_negative = np.append(y_data_negative, np.asarray(1, dtype='float32'))
                            else:
                                y_data_negative = np.append(y_data_negative, np.asarray(0, dtype='float32'))
                        if flag == 'p':
                            positive_music = np.append(positive_music, np.asarray(tempData, dtype='float32'))
                            if 'mdd' in filenames[i]:
                                y_data_positive = np.append(y_data_positive, np.asarray(1, dtype='float32'))
                            else:
                                y_data_positive = np.append(y_data_positive, np.asarray(0, dtype='float32'))
                        #print(negative_music.shape)
        filenames.remove(filenames[i])
        X_data = np.array(np.append(negative_music,positive_music))
        y_data = np.array(np.append(y_data_negative,y_data_positive))
        print(X_data.shape)
        print(y_data.shape)
        return negative_music, positive_music, y_data_negative, y_data_positive



def get_filenames():
    with open('C:\\Users\SHYAM\Documents\Minor Project\Data\participants.tsv') as csvfile:
        reader=csv.DictReader(csvfile,delimiter='\t')
        for row in reader:
            filenames.append(row['participant_id'])
    return filenames

filenames = get_filenames()
a,b,c,d = get_data(1,filenames)

im = image.smooth_img(imageMusicPath, fwhm=6)
                        print("Doesn't work")
                        print(im.shape)
                        mean_i = image.mean_img(im)
                        plot_epi(mean_i, title='Smoothed mean EPI',cut_coords=[-25,-37,-6])
                        plotting.show()