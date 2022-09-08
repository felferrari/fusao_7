import numpy as np
from ops.ops import load_json
import os
from sklearn.metrics import f1_score, average_precision_score
import pandas as pd

exp_name = 'lf_pm'

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
test_crop = conf['test_crop']
n_imgs = conf['n_imgs']

exp_path = os.path.join('exps', exp_name)
models_path = os.path.join(exp_path, 'models')
logs_path = os.path.join(exp_path, 'logs')
pred_path = os.path.join(exp_path, 'predicted')
visual_path = os.path.join(exp_path, 'visual')

prep_path = os.path.join('img', 'prepared')

label = (np.load(os.path.join(prep_path, f'label_2020.npy')) == 1).astype(np.uint8)

f1 = []
ap = []
for test_image in range(n_imgs):
    predicted = np.load(os.path.join(pred_path, f'pred_{test_image}.npy'))[:,:,1]

    f1_m = f1_score(label.flatten(), (predicted.flatten()>0.5).astype(np.uint8))

    ap_m = average_precision_score(label.flatten(), predicted.flatten())

    print(f'image {test_image}: f1Score:{f1_m}, AP:{ap_m}')

    f1.append(f1_m)
    ap.append(ap_m)

pd.DataFrame({
    'Image': list(range(n_imgs)),
    'F1 Score': f1,
    'AP': ap
}).to_excel(os.path.join(visual_path, f'metrics_{exp_name}.xlsx'))