import tensorflow as tf
from models.models import late_fusion_resunet
from models.model_t import SM_Transformer_PM
from ops.ops import load_json
import os
from models.losses import WBCE
from ops.dataloader_pm import PredictDataGen_opt
import numpy as np


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

exp_name = 'tr_sm_opt_pm'
exp_path = os.path.join('exps', exp_name)
models_path = os.path.join(exp_path, 'models')
logs_path = os.path.join(exp_path, 'logs')
pred_path = os.path.join(exp_path, 'predicted')

shape_opt = (patch_size, patch_size, n_opt_layers)
shape_sar = (patch_size, patch_size, n_sar_layers)
model_size = [32, 64, 128, 256]

for test_image in range(n_imgs):

    test_dataset = PredictDataGen_opt(2020, test_image)

    blocks_shape = test_dataset.blocks_shape
    img_shape = test_dataset.shape

    preds_l = []

    for model_idx in range(n_train_models):
        #model = tf.keras.models.load_model(os.path.join(models_path, f'model_{model_idx}'), compile=False)
        model = SM_Transformer_PM(n_classes, name = 'sm_opt')

        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = WBCE(class_weights)
        metrics = ['accuracy']

        model.compile(
            loss=loss,
            optimizer = optimizer,
            metrics = metrics
        )
        model.load_weights(os.path.join(models_path, f'model_{model_idx}'))
        
        pred = []
        for test_batch in test_dataset:
            pred_batch = model.predict_on_batch(test_batch)
            pred.append(pred_batch)

        pred = np.concatenate(pred, axis=0).reshape(blocks_shape+(patch_size, patch_size, n_classes))[: ,: ,test_crop:-test_crop ,test_crop:-test_crop, :]

        pred_reconstructed = None
        for line_i in pred:
            if pred_reconstructed is None:
                pred_reconstructed = np.column_stack(line_i)
            else:
                pred_reconstructed = np.row_stack((pred_reconstructed, np.column_stack(line_i)))

        pred_reconstructed = pred_reconstructed[:img_shape[0], :img_shape[1], :]
        preds_l.append(pred_reconstructed.astype(np.float16))
        #np.save(os.path.join(pred_path, f'pred_{test_image}_{model_idx}.npy'), pred_reconstructed)

    preds_l = np.array(preds_l)
    pred_mean = np.mean(preds_l, axis=0)
    np.save(os.path.join(pred_path, f'pred_{test_image}.npy'), pred_mean)
