import os
import scipy.io as io
from tqdm import tqdm
import numpy as np
import h5py
from data_utils import *

def save_data():
    print('### Load CUB data')
    print('current path:',os.getcwd())
    ''' path setting '''
    split_path = '/userhome/raw_data/zsl_data/APY'
    save_data_path = checkdir(os.path.join('./apy'))
    image_path = '/userhome/raw_data/apy/'

    ''' read img&att '''
    att_data = io.loadmat(checkfile(os.path.join(split_path, 'att_splits.mat')))
    image_data = io.loadmat(checkfile(os.path.join(split_path, 'res101.mat')))
    image_list = convert_mat_to_list(image_data['image_files'], 'apy')
    labels = image_data['labels'].squeeze().astype(int) - 1
    print('label:',labels.shape)
    
    ''' att '''
    all_att = att_data['att'].transpose()
    print('all_att:',all_att.shape)
    
    ''' image split '''
    train_id = att_data['trainval_loc'].squeeze() - 1
    print('train_id:',train_id.shape)
    test_unseen_id = att_data['test_unseen_loc'].squeeze() - 1
    print('test_unseen_id:',test_unseen_id.shape)
    test_seen_id = att_data['test_seen_loc'].squeeze() - 1
    print('test_seen_id:',test_seen_id.shape)
    
    ''' class split '''
    seen_class = np.unique(labels[train_id])
    print('seen_class:',seen_class)
    unseen_class = np.unique(labels[test_unseen_id])
    print('unseen_class:',unseen_class)
        
    ''' save '''
    save_path = checkdir(os.path.join('./apy'))
    h5_path = os.path.join(save_data_path, 'data_info.h5')

    if os.path.exists(h5_path):
        print "Skip store semantic features."
    else:
        h5_semantic_file = h5py.File(h5_path, 'w')
        # save classes split
        h5_semantic_file.create_dataset('seen_class', seen_class.shape, dtype=np.int32)
        h5_semantic_file.create_dataset('unseen_class', unseen_class.shape, dtype=np.int32)
        # save att
        h5_semantic_file.create_dataset('all_att', all_att.shape, dtype=np.float32)
        # image path

        h5_semantic_file['seen_class'][...] = seen_class
        h5_semantic_file['unseen_class'][...] = unseen_class
        h5_semantic_file['all_att'][...] = all_att
        h5_semantic_file['img_path'] = image_path

        h5_semantic_file.close()

    ''' write visual feats '''
    save_splits = ['train','test_seen','test_unseen']
    train_list = open(save_data_path+'/train.list','w') 
    test_seen_list = open(save_data_path+'/test_seen.list','w')
    test_unseen_list = open(save_data_path+'/test_unseen.list','w')    
    
    for each_save in save_splits:
        for i in eval(each_save+'_id'):
            eval(each_save+'_list').write('{} {} \n'.format(image_list[i],labels[i]))
        eval(each_save+'_list').close()

if __name__ == '__main__':
    save_data()
