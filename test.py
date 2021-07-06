from PIL.Image import ANTIALIAS
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import auc, roc_curve, roc_auc_score,f1_score,average_precision_score, precision_recall_curve

import config as c
from utils import *
from localization import export_gradient_maps
from train import Score_Observer
from model import DifferNet, load_weights
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from beautifultable import BeautifulTable
import pandas as pd
import seaborn as sns


train_set, test_set = load_datasets(c.dataset_path, c.class_name)
_, test_loader = make_dataloaders(train_set, test_set)

model = DifferNet()
model = load_weights(model, "/best.weights")
model.to(c.device)

model.eval()

test_z = list()
test_labels = list()
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar_test)):
        inputs, labels = preprocess_batch(data)
        z = model(inputs)
        test_z.append(z)
        test_labels.append(t2np(labels))

test_labels = np.concatenate(test_labels)
is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))

if c.save_hist:
    scores_dict = {}
    scores_dict['scores'] = anomaly_score
    scores_dict['labels'] = is_anomaly

    hist = pd.DataFrame.from_dict(scores_dict)

    # Filter normal and abnormal scores.
    abn_scr = hist.loc[hist.labels == 1]['scores']
    nrm_scr = hist.loc[hist.labels == 0]['scores']

    # Create figure and plot the distribution.
    sns.distplot(nrm_scr, label=r'Normal Scores', bins=20)
    sns.distplot(abn_scr, label=r'Abnormal Scores', bins=20)

    plt.legend()
    plt.yticks([])
    plt.xlabel(r'Anomaly Scores')
    plt.savefig('histogram.jpg')

if c.save_prc:
    precision, recall, thresholds = precision_recall_curve(
        is_anomaly, anomaly_score)
    prc_auc = auc(recall, precision)
    fig, ax = plt.subplots()
    ax.step(recall,precision,color='r',alpha=0.99,where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig("PRC.png")

if c.save_roc:
    fpr, tpr, _ = roc_curve(is_anomaly, anomaly_score)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
    plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
    plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join('./', "ROC.png"))
    plt.close()

median = np.median(anomaly_score)
print(median)
anomaly_score_binarized = [1 if x >= median else 0 for x in anomaly_score]

ras = roc_auc_score(is_anomaly, anomaly_score)
assert (roc_auc == ras)

result_table = BeautifulTable()
result_table.columns.header = ["Metric", "Result"]
result_table.rows.append(["F1-Score", f1_score(is_anomaly, anomaly_score_binarized)])
result_table.rows.append(["AUC-ROCs", roc_auc])
result_table.rows.append(["Average Precision", average_precision_score(is_anomaly, anomaly_score_binarized)])
result_table.rows.append(["AUC-PRC", prc_auc])
print(result_table) 

