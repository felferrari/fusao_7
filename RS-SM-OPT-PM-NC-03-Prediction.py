import tensorflow as tf
from ops.ops import load_json
from models.models import ResUnetPM, build_resunet
import os
from models.losses import WBCE, WFocal
from ops.dataloader_pm_nc_comp import PredictDataGen_opt, get_train_val_dataset, data_augmentation, prep_data_opt
#from ops.dataloader_pm_nc import PredictDataGen_opt, get_train_val_dataset, data_augmentation, prep_data_opt
import numpy as np
import time
import sys

conf = load_json(os.path.join('conf', 'conf.json'))
img_source = conf['img_source']
batch_size = conf['batch_size']
learning_rate = conf['learning_rate']
n_train_models = conf['n_train_models']
patch_size = conf['patch_size']
n_classes = conf['n_classes']
n_opt_layers = conf['n_opt_layers']
n_sar_layers = conf['n_sar_layers']
class_weights = conf['class_weights']
test_crop = conf['test_crop']
n_imgs = conf['n_imgs']
n_exps = conf['n_exps']

exp_name = 'exp_1'
exp_path = os.path.join('D:', 'Ferrari', 'exps_7', exp_name)

models_path = os.path.join(exp_path, 'models')
logs_path = os.path.join(exp_path, 'logs')
pred_path = os.path.join(exp_path, 'predicted')
visual_path = os.path.join(exp_path, 'visual')


shape_opt = (patch_size, patch_size, n_opt_layers)
shape_sar = (patch_size, patch_size, n_sar_layers)
shape_previous = (patch_size, patch_size, 1)

model_size = [64, 128, 256, 512]

outfile = os.path.join(visual_path, f'pred_{exp_name}.txt')

with open(outfile, 'w') as sys.stdout:

    test_dataset = PredictDataGen_opt(2019)
    ds_train, ds_val, n_patches_train, n_patches_val = get_train_val_dataset(2019)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds_train = ds_train.map(data_augmentation, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.map(prep_data_opt, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_val = ds_val.map(data_augmentation, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.map(prep_data_opt, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.prefetch(AUTOTUNE)

    train_steps = (n_patches_train // batch_size)
    val_steps = (n_patches_val // batch_size)

    blocks_shape = test_dataset.blocks_shape
    img_shape = test_dataset.shape

    preds_l = []

    for model_idx in range(n_exps):
        model = ResUnetPM(model_size, n_classes, name = f'{exp_name}_{model_idx}')
        print(model)

        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = WBCE(class_weights)
        metrics = ['accuracy']
        print(loss)

        model.compile(
            loss=loss,
            optimizer = optimizer,
            metrics = metrics
        )
        model.load_weights(os.path.join(models_path, f'model_{model_idx}'))

        model.evaluate(ds_train, verbose = 2, steps = train_steps)
        model.evaluate(ds_val, verbose = 2, steps = val_steps)

        pred = []
        t0 = time.perf_counter()
        for test_batch in test_dataset:
            pred_batch = model.predict_on_batch(test_batch)
            pred.append(pred_batch)

        print(f'Prediction time: {time.perf_counter() - t0}')

        pred = np.concatenate(pred, axis=0).reshape(blocks_shape+(patch_size, patch_size, n_classes))[: ,: ,test_crop:-test_crop ,test_crop:-test_crop, :]

        pred_reconstructed = None
        for line_i in pred:
            if pred_reconstructed is None:
                pred_reconstructed = np.column_stack(line_i)
            else:
                pred_reconstructed = np.row_stack((pred_reconstructed, np.column_stack(line_i)))

        pred_reconstructed = pred_reconstructed[:img_shape[0], :img_shape[1], :]
        #preds_l.append(pred_reconstructed.astype(np.float16))
        np.save(os.path.join(pred_path, f'pred_{model_idx}.npy'), pred_reconstructed.astype(np.float16))

    #preds_l = np.array(preds_l)
    #pred_mean = np.mean(preds_l, axis=0)
    #np.save(os.path.join(pred_path, f'pred_{test_image}.npy'), pred_mean)
