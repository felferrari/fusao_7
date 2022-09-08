import tensorflow as tf
from models.models import sm_resunet_pm
from ops.ops import load_json
import os
from models.losses import WBCE
from ops.dataloader_pm import get_train_val_dataset, data_augmentation, prep_data_opt


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

exp_name = 'sm_opt_pm'
exp_path = os.path.join('exps', exp_name)
models_path = os.path.join(exp_path, 'models')
logs_path = os.path.join(exp_path, 'logs')

shape_opt = (patch_size, patch_size, n_opt_layers)
shape_sar = (patch_size, patch_size, n_sar_layers)
shape_previous = (patch_size, patch_size, 1)
model_size = [32, 64, 128, 256]

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train, ds_val, n_patches_train, n_patches_val = get_train_val_dataset(2019)

ds_train = ds_train.map(data_augmentation, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.map(prep_data_opt, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.shuffle(50*batch_size)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_val = ds_val.map(data_augmentation, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.map(prep_data_opt, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.shuffle(50*batch_size)
ds_val = ds_val.batch(batch_size)
ds_val = ds_val.prefetch(AUTOTUNE)

train_steps = n_patches_train // batch_size
val_steps = n_patches_val // batch_size


for model_idx in range(n_train_models):
    model = sm_resunet_pm(shape_opt, shape_previous, model_size, n_classes, reg_weight = None, name = 'sm_opt')

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = WBCE(class_weights)
    metrics = ['accuracy']

    model.compile(
        loss=loss,
        optimizer = optimizer,
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

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(models_path, f'model_{model_idx}'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        save_freq='epoch',
    )

    callbacks = [
        early_stop,
        tensorboard,
        model_checkpoint
        ]

    
    history = model.fit(
        x=ds_train,
        validation_data=ds_val,
        epochs = max_epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=1, 
        callbacks = callbacks
    )


