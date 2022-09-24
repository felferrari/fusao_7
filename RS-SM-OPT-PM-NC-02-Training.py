import tensorflow as tf
from models.models import ResUnetPM, build_resunet
from ops.ops import load_json
import os
from models.losses import WBCE, WFocal
from ops.dataloader_pm_nc_comp import get_train_val_dataset, data_augmentation, prep_data_opt
#from ops.dataloader_pm_nc import get_train_val_dataset, data_augmentation, prep_data_opt
import sys
import time
from multiprocessing import Process
import numpy as np

exp_name = 'exp_1'

conf = load_json(os.path.join('conf', 'conf.json'))
img_source = conf['img_source']
max_epochs = conf['max_epochs']
batch_size = conf['batch_size']
learning_rate = conf['learning_rate']
n_train_models = conf['n_train_models']
patch_size = conf['patch_size']
n_classes = conf['n_classes']
n_opt_layers = conf['n_opt_layers']
n_sar_layers = conf['n_sar_layers']
class_weights = conf['class_weights']
train_patience = conf['train_patience']
n_exps = conf['n_exps']

exp_path = os.path.join('D:', 'Ferrari', 'exps_7', exp_name)

models_path = os.path.join(exp_path, 'models')
logs_path = os.path.join(exp_path, 'logs')
visual_path = os.path.join(exp_path, 'visual')

def run(model_idx):
    outfile = os.path.join(visual_path, f'train_{exp_name}_{model_idx}.txt')

    with open(outfile, 'w') as sys.stdout:

        shape_opt = (patch_size, patch_size, n_opt_layers)
        shape_sar = (patch_size, patch_size, n_sar_layers)
        shape_previous = (patch_size, patch_size, 1)

        model_size = [64, 128, 256, 512]

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        ds_train, ds_val, n_patches_train, n_patches_val = get_train_val_dataset(2018)

        ds_train = ds_train.map(data_augmentation, num_parallel_calls=AUTOTUNE)
        ds_train = ds_train.map(prep_data_opt, num_parallel_calls=AUTOTUNE)
        ds_train = ds_train.shuffle(50*batch_size)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.prefetch(AUTOTUNE)

        ds_val = ds_val.map(data_augmentation, num_parallel_calls=AUTOTUNE)
        ds_val = ds_val.map(prep_data_opt, num_parallel_calls=AUTOTUNE)
        ds_val = ds_val.batch(batch_size)
        ds_val = ds_val.prefetch(AUTOTUNE)

        train_steps = (n_patches_train // batch_size)
        val_steps = (n_patches_val // batch_size)

        print(f'Batch size: {batch_size}')
        print(f'Learning Rate: {learning_rate}')
        print(f'Class Weights: {class_weights}')

        model = ResUnetPM(model_size, n_classes, name = f'{exp_name}_{model_idx}')
        print(model)

        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = WBCE(class_weights)
        print(loss)

        metrics = ['accuracy']

        model.compile(
            loss=loss,
            optimizer = optimizer,
            #run_eagerly = True,
            metrics = metrics
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            restore_best_weights = True,
            patience=train_patience
        )

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(logs_path, f'log_tb_{model_idx}'),
            histogram_freq = 10
        )

        callbacks = [
            early_stop,
            tensorboard
            ]

        t0 = time.perf_counter()
        history = model.fit(
            x=ds_train,
            validation_data=ds_val,
            epochs = max_epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            verbose=2, 
            callbacks = callbacks
        )
        print(f'Training time: {(time.perf_counter() - t0)/60} mins')
        model.summary()
        model.save_weights(os.path.join(models_path, f'model_{model_idx}'))


if __name__=="__main__":
    conf = load_json(os.path.join('conf', 'conf.json'))
    n_exps = conf['n_exps']

    
    for model_idx in range(n_exps):
        p = Process(target=run, args=(model_idx,))
        p.start()
        p.join()
