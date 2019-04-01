# python gabor.py [input_dir] [output_dir] [#orientation] [#scale] [doEqual]
import os
import sys
import re
import numpy as np
import cv2
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
sys.path.append('../lib')
#from readMatToImg import readMatToImg

ANT_PAIR = 6

#25people
action_idx  = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13,
               'n':14, 'o':15, 'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25}

#action_idx  = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8}

#20people
#action_idx  = {'a':1, 'b':2, 'c':3, 'd':4, 'f':6, 'h':8, 'i':9, 'j':10, 'l':12, 'm':13, 'n':14, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25}

#15people
#action_idx  = {'a':1, 'b':2, 'c':3, 'd':4, 'f':6, 'h':8, 'i':9, 'j':10, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'y':25}

#10people
#action_idx  = {'a':1, 'b':2, 'c':3, 'h':8, 'i':9, 'q':17, 'r':18, 's':19, 't':20, 'u':21}

#5people
#action_idx  = {'i':9, 'q':17, 's':19, 't':20, 'u':21}


def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def build_filters(num_scale, num_orien):
    filters = []
    ksize = 15
    gamma = 1
    scale_step = 2.1/num_scale  # 3.1 - 1 = 2.0
    for theta in np.arange(0, np.pi, np.pi / num_orien):
        for sigma in np.arange(1, 3.1, scale_step):
            lamda = sigma + 1
            #print 'ksize = ', ksize, ', gamma = ', gamma, 'theta = ', theta, 'sigma = ', sigma, 'lambda = ', lamda
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)

    # Append more vertical filters
    #vksize = 17
    #vtheta = 0
    #for sigma in np.arange(1, 3.1, 0.4):
    #    lamda = sigma + 1
    #    kern = cv2.getGaborKernel((vksize, vksize), sigma, vtheta, lamda, gamma, 0, ktype=cv2.CV_32F)
    #    kern /= 1.5 * kern.sum()
    #    filters.append(kern)

    return filters

def process(img, one_filter):
    accum = np.zeros_like(img)
    fimg = cv2.filter2D(img, -1, one_filter)
    #accum = np.maximum(accum, fimg)
    np.maximum(accum, fimg, accum)

    return accum

def main():
    # Read in 4_jpg/ (each directory corresponding to a training data)
    input_infor = ['../Result/Denoise', '../Result/Feature_Extraction', 8, 6, 0]

    if len(input_infor) != 6:
        print 'Usage: python gabor.py input_dir output_dir num_orien num_scale doHistEq'


    input_dir  = input_infor[0]
    basename = input_dir.split('/')
    if basename[-1] == '':
        basename = basename[-2]
    else:
        basename = basename[-1]

    #if not os.path.isdir(os.path.join('filtered_map', basename)):
    #    os.mkdir(os.path.join('filtered_map', basename))

    output_dir = input_infor[1]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    num_orien  = int(input_infor[2])
    num_scale  = int(input_infor[3])
    doHistEqual= int(input_infor[4])

    output_file_list = []
    for i in xrange(0, ANT_PAIR):
        outf = open(output_dir + '/' + str(i + 1) + '_feat', 'w')
        output_file_list.append(outf)

    # Read in every channel for each training data
    # EX: directory = 'jog8', image_file = '3.jpg'
    dir_list = os.listdir(input_dir)
    for directory in dir_list:
        # Do not process folder of original CSIs
        if directory == 'orig':
            continue

        # Create image directory if not exists
        #img_dir = os.path.join('filtered_map', basename, directory)
        #if not os.path.isdir(img_dir):
        #    os.mkdir(img_dir)

        # Remove empty
        #cate = re.search('([a-z]+)\d+', directory).group(1)
        #if cate == 'empty': continue

        print 'Processing ' + directory + ' ...'
        file_list = os.listdir(input_dir + '/' + directory)

        # Get gabor filters
        filters = []
        filters = build_filters(num_scale, num_orien)
        filters = np.asarray(filters)

        #fig = plt.figure()
        '''
        for i in xrange(0, len(filters)):
            ax = fig.add_subplot(1, len(filters), i)
            plt.imshow(filters[i], cmap=mpl.cm.gray)
        '''
        '''
        for i in xrange(0, len(filters)):
            plt.imshow(filters[i], cmap=mpl.cm.gray)
            plt.savefig("filters_svg/filter" + str(i) + ".svg", format="svg")
        print 'svg finish'
        raw_input()
        '''

        #numChannel = len(file_list)
        #step = 1.0/numChannel
        for idx, image_file in enumerate(file_list):
            if os.path.splitext(image_file)[0] == 'room':
                continue
            feature = []

            # Read in img and convert to gray-scale
            img = cv2.imread(input_dir + '/' + directory + '/' + image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Decide if need histequal
            if doHistEqual:
                img = cv2.equalizeHist(img)
            #if not os.path.isdir(output_dir + '/' + directory):
            #    os.mkdir(output_dir + '/' + directory)
            #cv2.imwrite(output_dir + '/' + directory + '/' + image_file, img)

            # Apply filters on img
            res = []
            for i in xrange(len(filters)):
                res1 = process(img, filters[i])
                res.append(np.asarray(res1))
            ''' 
            #fig.subplots_adjust(left=0, bottom=1-(idx+1)*step, right=1, top=1-idx*step, hspace=0)
            #ax = fig.add_subplot(numChannel, 1, idx+1)
            ax = fig.add_axes([0, 1-(idx+1)*step, 1, step])

            ax.axis('off')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

            plt.imshow(res[0], cmap=mpl.cm.gray, aspect='auto')

            plt.savefig(os.path.join(img_dir, 'res.jpg'), pad_inches=0)
            plt.close()
            '''
            '''
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            plt.imshow(img, cmap=mpl.cm.gray)
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(filters[8], cmap=mpl.cm.gray)
            ax = fig.add_subplot(1, 3, 3)
            plt.imshow(res[8], cmap=mpl.cm.gray)
            plt.show()
            '''
            # Calculate mean, std for each filtered result
            for i in xrange(len(res)):
                mean = np.mean(res[i])
                std = np.std(res[i])
                feature.append(mean)
                feature.append(std)

            # Get ant_pair_index for i_feat
            ant_pair_index, ext = os.path.splitext(image_file)
            ant_pair_index = int(ant_pair_index) - 1
            # Write data name
            output_file_list[ant_pair_index].write('%s ' % directory)

            # Write features of each channel
            #feature[0:len(feature):2] = feature[0:len(feature):2] / sum(feature[0:len(feature):2])
            for f in feature:
                output_file_list[ant_pair_index].write('%s ' % f)

            # Write label
            label = re.search('([\D]+)\d+', directory).group(1)
            label = action_idx.get(label, '-1')
            if label == '-1':
                continue
            output_file_list[ant_pair_index].write('%s\n' % label)

if __name__ == '__main__':
    main()
