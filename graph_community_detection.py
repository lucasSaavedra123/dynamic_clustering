from LocalizationClassifier import LocalizationClassifier
from CONSTANTS import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


#Training
classifier = LocalizationClassifier(10,10)
classifier.fit_with_datasets_from_path('./datasets_shuffled/train')
classifier.save_model()
#Testing
"""
classifier = LocalizationClassifier(10,10)
classifier.load_model()
classifier.test_with_datasets_from_path('./datasets_shuffled/train', apply_threshold=False, save_result=True)
"""
#Visualize Testing
# normalized = True
# result = pd.read_csv('result.csv')

# ground_truth = result['true'].values.tolist()
# Y_predicted = result['pred'].values.tolist()

# confusion_mat = confusion_matrix(y_true=ground_truth, y_pred=Y_predicted)

# if normalized:
#     confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]

# labels = ["Non-Clusterized", "Clusterized"]

# confusion_matrix_dataframe = pd.DataFrame(data=confusion_mat, index=labels, columns=labels)
# sns.set(font_scale=1.5)
# color_map = sns.color_palette(palette="Blues", n_colors=7)
# sns.heatmap(data=confusion_matrix_dataframe, annot=True, annot_kws={"size": 15}, cmap=color_map)

# plt.title(f'Confusion Matrix')
# plt.rcParams.update({'font.size': 15})
# plt.ylabel("Ground truth", fontsize=15)
# plt.xlabel("Predicted label", fontsize=15)
# plt.show()

#Infering
#classifier = LocalizationClassifier(10,10)
#classifier.load_model()

#['CDx_mAb', 'CDx_BTX', 'CDx-Chol_BTX', 'CDx-Chol_mAb', 'Control_BTX', 'Control_mAb']

#for experimental_conditions in ['Control_BTX', 'Control_mAb']:
#    classifier.transform_magik_dataframe_to_smlm_dataset(classifier.predict(classifier.get_dataset_from_path(f'./data/{experimental_conditions}.csv'))).to_csv(f'./data/{experimental_conditions}_localization_classifier_result.csv')


"""
from plot_metric.functions import BinaryClassification
import pandas as pd

resultados = pd.read_csv('result_100.csv')
y_test = resultados['true'].values.tolist()
y_pred = resultados['pred'].values.tolist()

# Visualisation with plot_metric
bc = BinaryClassification(y_test, y_pred, labels=["Non-Clusterized", "Clusterized"])

# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve(threshold=0.725)
plt.show()
"""