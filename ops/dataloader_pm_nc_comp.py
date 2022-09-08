import numpy as np
import tensorflow as tf
import os
from ops.ops import load_json
from skimage.util import view_as_windows
import random

def train_data_gen(year):
    conf = load_json(os.path.join('conf', 'conf.json'))
    prep_path = os.path.join('img', 'prepared')

    n_opt_layers = conf['n_opt_layers']
    n_sar_layers = conf['n_sar_layers']
    patch_size = conf['patch_size']
    n_classes = conf['n_classes']

    patches_idxs_train = np.load(os.path.join(prep_path, f'patches_{year}_train.npy' ))
    patches_idxs_val = np.load(os.path.join(prep_path, f'patches_{year}_val.npy' ))

    c_patches_idxs_train = np.load(os.path.join(prep_path, f'c_patches_{year}_train.npy' ))
    c_patches_idxs_val = np.load(os.path.join(prep_path, f'c_patches_{year}_val.npy' ))

    n_patches_idxs_train = c_patches_idxs_train.shape[0]
    n_patches_idxs_val = c_patches_idxs_val.shape[0]
    
    t_0 = f'{year-1}'
    t_1 = f'{year}'
    opt_0 = []
    opt_1 = []

    labels = np.load(os.path.join(prep_path, f'label_{year}.npy')).reshape((-1,1))
    previous = np.load(os.path.join(prep_path, f'previous_{year}.npy')).reshape((-1,1))

    opt_0 = np.load(os.path.join(prep_path, f'opt_{t_0}.npy')).reshape((-1,n_opt_layers))
    opt_1 = np.load(os.path.join(prep_path, f'opt_{t_1}.npy')).reshape((-1,n_opt_layers))

    def func_train():
        while True:
            np.random.shuffle(patches_idxs_train)
            for patch_idx in patches_idxs_train:
                yield ( #yield a patch wiht at least 2% of deforestation
                    opt_0[patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    opt_1[patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    previous[patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    tf.keras.utils.to_categorical(labels[patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                )
                c_patch_idx = c_patches_idxs_train[random.randrange(n_patches_idxs_train)]
                yield ( #yield a patch with less than 2% of deforestation
                    opt_0[c_patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    opt_1[c_patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    previous[c_patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    tf.keras.utils.to_categorical(labels[c_patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                )
    def func_val():
        while True:
            for patch_idx in patches_idxs_val:
                yield (
                    opt_0[patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    opt_1[patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    previous[patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    tf.keras.utils.to_categorical(labels[patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                    )
                c_patch_idx = c_patches_idxs_val[random.randrange(n_patches_idxs_val)]
                yield (
                    opt_0[c_patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    opt_1[c_patch_idx].reshape((patch_size,patch_size,n_opt_layers)).astype(np.float32),
                    previous[c_patch_idx].reshape((patch_size,patch_size,1)).astype(np.float32),
                    tf.keras.utils.to_categorical(labels[c_patch_idx].reshape((patch_size, patch_size, 1)), n_classes).astype(np.float32)
                )
                    

    return func_train, func_val, 2*patches_idxs_train.shape[0], 2*patches_idxs_val.shape[0]

def get_train_val_dataset(year):
    conf = load_json(os.path.join('conf', 'conf.json'))

    n_opt_layers = conf['n_opt_layers']
    patch_size = conf['patch_size']
    n_classes = conf['n_classes']

    output_signature = (
        tf.TensorSpec(shape=(patch_size , patch_size , n_opt_layers), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size , patch_size , n_opt_layers), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size , patch_size , 1), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size , patch_size , n_classes), dtype=tf.float32),
        )
    data_gen_train, data_gen_val, n_patches_train, n_patches_val = train_data_gen(year)
    ds_train = tf.data.Dataset.from_generator(
        generator = data_gen_train, 
        output_signature = output_signature
        )

    ds_val = tf.data.Dataset.from_generator(
        generator = data_gen_val, 
        output_signature = output_signature
        )
    return ds_train, ds_val, n_patches_train, n_patches_val

def data_augmentation(*data):
    x_0 = data[0] 
    x_1 = data[1] 
    x_2 = data[2] 
    x_3 = data[3] 
    if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
        x_0 = tf.image.flip_left_right(x_0)
        x_1 = tf.image.flip_left_right(x_1)
        x_2 = tf.image.flip_left_right(x_2)
        x_3 = tf.image.flip_left_right(x_3)
    if tf.math.greater(tf.random.uniform(shape=[], minval=0, maxval=1), tf.constant(0.5)):
        x_0 = tf.image.flip_up_down(x_0)
        x_1 = tf.image.flip_up_down(x_1)
        x_2 = tf.image.flip_up_down(x_2)
        x_3 = tf.image.flip_up_down(x_3)

    k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    x_0 = tf.image.rot90(x_0, k)
    x_1 = tf.image.rot90(x_1, k)
    x_2 = tf.image.rot90(x_2, k)
    x_3 = tf.image.rot90(x_3, k)

    return x_0, x_1, x_2, x_3

def prep_data(*data):
    return (
        (data[0], data[1], data[2]), data[3]
    )

def prep_data_opt(*data):
    return (
        (data[0], data[1], data[2]), data[3]
    )


class PredictDataGen(tf.keras.utils.Sequence):
    def __init__(self, year, idx):
        conf = load_json(os.path.join('conf', 'conf.json'))
        prep_path = os.path.join('img', 'prepared')

        n_opt_layers = conf['n_opt_layers']
        n_sar_layers = conf['n_sar_layers']
        patch_size = conf['patch_size']
        n_classes = conf['n_classes']
        test_crop = conf['test_crop']
        self.batch_size = conf['batch_size']
        crop_size = patch_size - 2*test_crop

        t_0 = f'{year-1}'
        t_1 = f'{year}'

        shape = np.load(os.path.join(prep_path, f'label_{year}.npy')).shape

        #n_shape = (shape[0]+2*test_crop , shape[1]+2*test_crop)

        pad_0 = crop_size - (shape[0] % crop_size)
        pad_1 = crop_size - (shape[1] % crop_size)

        pad_matrix = (
            (test_crop, test_crop + pad_0),
            (test_crop, test_crop + pad_1),
            (0,0)
        )

        n_shape = (shape[0] + pad_0 + 2*test_crop, shape[1] + pad_1 + 2*test_crop)

        self.opt_0 = np.pad(np.load(os.path.join(prep_path, f'opt_{t_0}_{idx}.npy')), pad_matrix, mode='reflect').reshape((-1,n_opt_layers))
        self.opt_1 = np.pad(np.load(os.path.join(prep_path, f'opt_{t_1}_{idx}.npy')), pad_matrix, mode='reflect').reshape((-1,n_opt_layers))

        self.sar_0 = np.pad(np.load(os.path.join(prep_path, f'sar_{t_0}_{idx}.npy')), pad_matrix, mode='reflect').reshape((-1,n_sar_layers))
        self.sar_1 = np.pad(np.load(os.path.join(prep_path, f'sar_{t_1}_{idx}.npy')), pad_matrix, mode='reflect').reshape((-1,n_sar_layers))

        self.previous = np.pad(np.expand_dims(np.load(os.path.join(prep_path, f'previous_{year}.npy')), axis=-1), pad_matrix, mode='reflect').reshape((-1,1))

        idx_matrix = np.arange(n_shape[0]*n_shape[1]).reshape(n_shape)

        idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), crop_size)

        self.blocks_shape = idx_patches.shape[0:2]
        self.shape = shape

        self.idx_patches = idx_patches.reshape((-1, patch_size, patch_size))

    def __len__(self):
        return 1 + (self.idx_patches.shape[0] // self.batch_size)

    def __getitem__(self, index):
        sel_idx_patches = self.idx_patches[index*self.batch_size:(index+1)*self.batch_size, :,:]
        return (
            self.opt_0[sel_idx_patches],
            self.opt_1[sel_idx_patches],
            self.sar_0[sel_idx_patches],
            self.sar_1[sel_idx_patches],
            self.previous[sel_idx_patches]

        )


class PredictDataGen_opt(tf.keras.utils.Sequence):
    def __init__(self, year):
        conf = load_json(os.path.join('conf', 'conf.json'))
        prep_path = os.path.join('img', 'prepared')

        n_opt_layers = conf['n_opt_layers']
        n_sar_layers = conf['n_sar_layers']
        patch_size = conf['patch_size']
        n_classes = conf['n_classes']
        test_crop = conf['test_crop']
        self.batch_size = conf['batch_size']
        crop_size = patch_size - 2*test_crop

        t_0 = f'{year-1}'
        t_1 = f'{year}'

        shape = np.load(os.path.join(prep_path, f'label_{year}.npy')).shape

        #n_shape = (shape[0]+2*test_crop , shape[1]+2*test_crop)

        pad_0 = crop_size - (shape[0] % crop_size)
        pad_1 = crop_size - (shape[1] % crop_size)

        pad_matrix = (
            (test_crop, test_crop + pad_0),
            (test_crop, test_crop + pad_1),
            (0,0)
        )

        n_shape = (shape[0] + pad_0 + 2*test_crop, shape[1] + pad_1 + 2*test_crop)

        self.opt_0 = np.pad(np.load(os.path.join(prep_path, f'opt_{t_0}.npy')), pad_matrix, mode='reflect').reshape((-1,n_opt_layers))
        self.opt_1 = np.pad(np.load(os.path.join(prep_path, f'opt_{t_1}.npy')), pad_matrix, mode='reflect').reshape((-1,n_opt_layers))

        self.previous = np.pad(np.expand_dims(np.load(os.path.join(prep_path, f'previous_{year}.npy')), axis=-1), pad_matrix, mode='reflect').reshape((-1,1))

        idx_matrix = np.arange(n_shape[0]*n_shape[1]).reshape(n_shape)

        idx_patches = view_as_windows(idx_matrix, (patch_size, patch_size), crop_size)

        self.blocks_shape = idx_patches.shape[0:2]
        self.shape = shape

        self.idx_patches = idx_patches.reshape((-1, patch_size, patch_size))

    def __len__(self):
        return 1 + (self.idx_patches.shape[0] // self.batch_size)

    def __getitem__(self, index):
        sel_idx_patches = self.idx_patches[index*self.batch_size:(index+1)*self.batch_size, :,:]
        return (
            self.opt_0[sel_idx_patches], 
            self.opt_1[sel_idx_patches], 
            self.previous[sel_idx_patches]
        )