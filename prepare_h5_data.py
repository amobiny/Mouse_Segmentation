# read envi file and create a h5 file
import envi
import scipy.misc
import h5py
import numpy as np


def load_data(envifile, stainfile, trainmask="", normalize = True):
    if trainmask == "":
        E = envi.envi(filename=envifile)
    else:
        mask = scipy.misc.imread(trainmask, flatten=True)
        E = envi.envi(envifile, mask=mask)
    F= E.loadall()    # return a matrix from envi file #bands * w * h
    F = np.rollaxis(F, 0, 3)

    # read the stained file
    stain = scipy.misc.imread(stainfile)
    if normalize:
        F, stain = norm(F, stain, mode='standard')

    else:

        stain = stain/255.

    return F , stain

def main():
    envifile_path = 'D:\\python-projects\\cnn-codes\\data\\DAPI\\1\\new\\'#"D:\\python-projects\\cnn-codes\\data\\davar\\a_train_c_test\\new\\a_train_b_valid_c_test\\"
    stainfile_path ='D:\\python-projects\\cnn-codes\\data\\DAPI\\1\\new\\' #"D:\\python-projects\\cnn-codes\\data\\davar\\a_train_c_test\\new\\a_train_b_valid_c_test\\"
    save_path= 'D:\\python-projects\\cnn-codes\\data\\DAPI\\1\\new\\'#'D:\\python-projects\\cnn-codes\\data\\davar\\a_train_c_test\\new\\a_train_b_valid_c_test\\'
    h5f = h5py.File(save_path + 'DAPI_kid_crop.h5', 'w')
    #train data
    envi_name = 'mosaic_base_bip_pca_mask2'#'a_mosaic_base_bip_pca60' #'ab_mosaic_base_bip_norm_pca60'
    target_name= 'target2.bmp'
    envifile = envifile_path + envi_name
    stainfile= stainfile_path + target_name
    E, stain = load_data(envifile, stainfile)
    x_train = E[np.newaxis]
    y_train = stain[np.newaxis]
    h5f.create_dataset('x_train', data=x_train)
    h5f.create_dataset('y_train', data=y_train)

    # validation data
    envi_name = 'mosaic_base_bip_pca_mask2_crop_test'
    target_name = 'cnn_re_crop.bmp'
    envifile = envifile_path + envi_name
    stainfile = stainfile_path + target_name
    E, stain = load_data(envifile, stainfile)
    h5f.create_dataset('x_valid', data=E)
    h5f.create_dataset('y_valid', data=stain)

    # #test data
    # envi_name = 'mosaic_base_bip_mask_pca_part'
    # target_name = 'm_brain_acu_neun_washed_reg1.bmp'
    # envifile = envifile_path + envi_name
    # stainfile = stainfile_path + target_name
    # E, stain = load_data(envifile, stainfile)
    # h5f.create_dataset('x_test', data=E)
    # h5f.create_dataset('y_test', data=stain)

    h5f.close()

    #read h5
    h5f = h5py.File(save_path + 'DAPI_kid_crop.h5', 'r')
    a= h5f['x_valid'][:]
    b = h5f['y_valid'][:]
    print(a.shape)
    print(b.shape)
    h5f.close()




def norm(x, y, mode):
    if mode == 'gaussian':
        pass
    elif mode == 'standard':
        x_max = np.array([np.max(x[:, :, i]) for i in range(x.shape[-1])])
        x_min = np.array([np.min(x[:, :, i]) for i in range(x.shape[-1])])
        x_norm = (x-x_min)/(x_max-x_min)
        y_norm = y/255.
        return x_norm, y_norm
if __name__ == '__main__':
    main()
