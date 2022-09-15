import numpy as np
from ops.ops import load_json, save_json
import os
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
import pandas as pd
from skimage.morphology import area_opening

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

exp_name = 'rs_sm_opt_pm_nc_5'
exp_path = os.path.join('D:', 'Ferrari', 'exps_7', exp_name)

models_path = os.path.join(exp_path, 'models')
logs_path = os.path.join(exp_path, 'logs')
pred_path = os.path.join(exp_path, 'predicted')
visual_path = os.path.join(exp_path, 'visual')

prep_path = os.path.join('img', 'prepared')


def conf_matrix(label, pred_prob, max_cloud_map):
    pred = np.zeros_like(pred_prob, dtype=np.uint8)
    pred[pred_prob >= 0.5] = 1
    #pred[pred_label == 1] = 1
    pred_removed = pred -  area_opening(pred, 625)
    label[pred_removed == 1] = 2

    keep_idx = label.flatten() != 2

    pred_eval = pred.flatten()[keep_idx]
    label_eval = label.flatten()[keep_idx]
    cmap_eval = max_cloud_map.flatten()[keep_idx]

    #mcp_idx
    low_mcp_idx = np.argwhere(cmap_eval <= 5).squeeze()
    med_mcp_idx = np.argwhere(np.logical_and(cmap_eval > 5, cmap_eval <= 70)).squeeze()
    hig_mcp_idx = np.argwhere(cmap_eval > 70).squeeze()

    low_tn, low_fp, low_fn, low_tp = confusion_matrix(label_eval[low_mcp_idx], pred_eval[low_mcp_idx], labels=[0,1]).ravel()
    med_tn, med_fp, med_fn, med_tp = confusion_matrix(label_eval[med_mcp_idx], pred_eval[med_mcp_idx], labels=[0,1]).ravel()
    hig_tn, hig_fp, hig_fn, hig_tp = confusion_matrix(label_eval[hig_mcp_idx], pred_eval[hig_mcp_idx], labels=[0,1]).ravel()

    return (
        (low_tn, low_fp, low_fn, low_tp),
        (med_tn, med_fp, med_fn, med_tp),
        (hig_tn, hig_fp, hig_fn, hig_tp)
    )



label = np.load(os.path.join(prep_path, f'label_2019.npy'))

low_m = np.zeros(4)
med_m = np.zeros(4)
hig_m = np.zeros(4)
glo_m = np.zeros(4)
for test_image in range(1):
    predicted = np.load(os.path.join(pred_path, f'pred_{test_image}.npy'))[:,:,1]

    max_cloud_map =  np.maximum(
        np.load(os.path.join(prep_path, f'cmap_2018.npy')),
        np.load(os.path.join(prep_path, f'cmap_2019.npy'))
    )
    

    low_data, med_data, hig_data = conf_matrix(label, predicted, max_cloud_map)

    low_m += np.array(low_data)
    med_m += np.array(med_data)
    hig_m += np.array(hig_data)
    glo_m += np.array(hig_data) + np.array(med_data) + np.array(low_data)

cloud = ['Low', 'Med', 'High', 'Global']

prec = [
    low_m[3] / (low_m[3] + low_m[1]),
    med_m[3] / (med_m[3] + med_m[1]),
    hig_m[3] / (hig_m[3] + hig_m[1]),
    glo_m[3] / (glo_m[3] + glo_m[1]),
]

recall = [
    low_m[3] / (low_m[3] + low_m[2]),
    med_m[3] / (med_m[3] + med_m[2]),
    hig_m[3] / (hig_m[3] + hig_m[2]),
    glo_m[3] / (glo_m[3] + glo_m[2]),
]

f1 = [
    2 * prec[0] * recall[0] / (prec[0] + recall[0]),
    2 * prec[1] * recall[1] / (prec[1] + recall[1]),
    2 * prec[2] * recall[2] / (prec[2] + recall[2]),
    2 * prec[3] * recall[3] / (prec[3] + recall[3])
]

acc = [
    (low_m[3] + low_m[0]) / (low_m[0] + low_m[1] + low_m[2] + low_m[3]),
    (med_m[3] + med_m[0]) / (med_m[0] + med_m[1] + med_m[2] + med_m[3]),
    (hig_m[3] + hig_m[0]) / (hig_m[0] + hig_m[1] + hig_m[2] + hig_m[3]),
    (glo_m[3] + glo_m[0]) / (glo_m[0] + glo_m[1] + glo_m[2] + glo_m[3]),
]

tn = [
    low_m[0],
    med_m[0],
    hig_m[0],
    glo_m[0],
]
fp = [
    low_m[1],
    med_m[1],
    hig_m[1],
    glo_m[1],
]
fn = [
    low_m[2],
    med_m[2],
    hig_m[2],
    glo_m[2],
]
tp = [
    low_m[3],
    med_m[3],
    hig_m[3],
    glo_m[3],
]

pd.DataFrame({
    'Cloud': cloud,
    'F1 Score': f1,
    'Precision': prec,
    'Recall': recall,
    'Accuracy':acc,
    'TP':tp,
    'FP':fp,
    'TN':tn,
    'FN':fn
}).to_excel(os.path.join(visual_path, f'metrics_{exp_name}.xlsx'))







