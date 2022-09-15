import numpy as np
from ops.ops import load_json, save_json
import os
from sklearn.metrics import f1_score, average_precision_score, confusion_matrix
import pandas as pd
from skimage.morphology import area_opening
from itertools import repeat
from multiprocessing import Pool, freeze_support
import matplotlib.pyplot as plt

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

label = np.load(os.path.join(prep_path, f'label_2019.npy'))

def prec_recall_curve(threshold, pred_prob, label):
    pred_bin = np.zeros_like(pred_prob, dtype=np.uint8)
    pred_bin[pred_prob >= threshold] = 1
    pred_removed = pred_bin -  area_opening(pred_bin, 625)
    label[pred_removed == 1] = 2
    del pred_removed

    keep_idx = label.flatten() != 2

    pred_eval = pred_bin.flatten()[keep_idx]
    label_eval = label.flatten()[keep_idx]

    del keep_idx

    tn, fp, fn, tp = confusion_matrix(label_eval, pred_eval).ravel()

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)

    return recall, prec, acc

def complete_nan_values(curve):
    vec_prec = curve[:,0]
    for j in reversed(range(len(vec_prec))):
        if np.isnan(vec_prec[j]):
            vec_prec[j] = vec_prec[j+1]

    vec_rec = curve[:,1]
    for j in (range(len(vec_rec))):
        if np.isnan(vec_rec[j]):
            vec_rec[j] = vec_rec[j-1]
    curve[:,1] = vec_rec
    return curve 

for test_image in range(1):
    predicted = np.load(os.path.join(pred_path, f'pred_{test_image}.npy'))[:,:,1]

    #recall, prec, acc = prec_recall_curve(predicted, label, 0.5)

    thresholds = np.linspace(predicted.min(), predicted.max(), num=50)

    if __name__=="__main__":
        freeze_support()
        with Pool(processes=7) as pool:
            curve = pool.starmap(prec_recall_curve, zip(thresholds, repeat(predicted), repeat(label)))
            curve = np.array(curve)
            curve = complete_nan_values(curve)
            prec = curve[:,0]
            recall = curve[:,1]

            #recall = np.insert(recall, 0, 0)
            #prec = np.insert(prec, 0, prec[0])
            #deltaR = recall[1:]-recall[:-1]
            #AP = np.sum(prec[1:]*deltaR)

            recall_ = np.insert(recall, 0, 0)
            prec_ = np.insert(prec, 0, prec[0])
            deltaR = recall_[1:]-recall_[:-1]
            m_prec_ = (prec_[1:] + prec_[:-1])/2
            AP = np.sum(m_prec_*deltaR)

            plt.figure(figsize=(10,10))
            plt.plot(curve[:,1],curve[:,0], 'b-', label = f'FUSION (AP: {AP:.4f})')
            plt.legend(loc="lower left")
            ax = plt.gca()
            ax.set_ylim([0,1.01])
            ax.set_xlim([0,1.01])
            plt.grid()
            plt.savefig(os.path.join(visual_path, f'result_{exp_name}.png'))

            np.save(os.path.join(visual_path, f'curve_{test_image}.npy'), curve)
        







    

