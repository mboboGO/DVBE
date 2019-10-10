import os
#import cv2
import numpy as np
#import skimage.io
import random


def convert_mat_to_list(mat, dataset, convert_type='simple'):
    ret_list = []
    if dataset == 'cub':
        for i in mat:
            ret_list.append(os.path.basename(i[0][0]))
    elif dataset == 'sun':
        if convert_type == 'simple':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/')[8:])))
        elif convert_type == 'origin':
            for i in mat:
                ret_list.append(str(i[0][0]))
                # import pdb; pdb.set_trace()
    elif dataset == 'awa2':
        if convert_type == 'origin':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/'))))
        elif convert_type == 'simple':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/')[-2:])))
    elif dataset == 'apy':
        if convert_type == 'simple':
            for i in mat:
                i_part = i[0][0].split('/')
                if len(i_part) == 12: # VOC image
                    ret_list.append(str("/".join(i[0][0].split('/')[8:])))
                elif len(i_part) == 9: #aYahoo image
                    ret_list.append(str("/".join(i[0][0].split('/')[-2:])))
                else:
                    raise NotImplementedError
    else:
        raise NotImplementedError

    return ret_list

    
def checkfile(datapath):
    assert os.path.exists(datapath), 'This is no file %s'%(datapath)
    return datapath

def checkdir(datapath):
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    return datapath

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def norm_feat(data):
    """
    e.g. CUB class attr size: 200 x 312
    """
    assert data.ndim == 2
    data_len = np.linalg.norm(data, axis=1)
    data_len += 1e-8
    norm_data = data / data_len[:, None]
    return norm_data

