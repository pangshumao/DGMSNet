import numpy as np
import os
import matplotlib.pyplot as plt



if __name__ == '__main__':
    data_dir = '/home/pangshumao/data/Spine_Localization_PIL'
    identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection-mv'
    name_list = ['VBs', 'IVDs', 'All']
    data = np.load(os.path.join(data_dir, 'result', identifier, 'all_evaluate_metrics.npz'))

    metrics = ['DSC', 'Precision', 'Recall']

    for metric in metrics:
        scores = []
        if metric == 'DSC':
            subject_ver_dices = data['subject_ver_dices']
            subject_ivd_dices = data['subject_ivd_dices']
            subject_dices = data['subject_dices']
            title = 'DSC (%)'
        elif metric == 'Precision':
            subject_ver_dices = data['subject_ver_precisions']
            subject_ivd_dices = data['subject_ivd_precisions']
            subject_dices = data['subject_precisions']
            title = 'Precision (%)'
        elif metric == 'Recall':
            subject_ver_dices = data['subject_ver_recalls']
            subject_ivd_dices = data['subject_ivd_recalls']
            subject_dices = data['subject_recalls']
            title = 'Recall (%)'

        subject_dices = np.where(subject_dices == -1.0, np.nan, subject_dices)
        subject_ver_dices = np.where(subject_ver_dices == -1.0, np.nan, subject_ver_dices)
        subject_ivd_dices = np.where(subject_ivd_dices == -1.0, np.nan, subject_ivd_dices)

        subject_dices = subject_dices[~np.isnan(subject_dices)] * 100
        subject_ver_dices = subject_ver_dices[~np.isnan(subject_ver_dices)] * 100
        subject_ivd_dices = subject_ivd_dices[~np.isnan(subject_ivd_dices)] * 100

        scores.append(subject_ver_dices)
        scores.append(subject_ivd_dices)
        scores.append(subject_dices)

        # plt.boxplot(DSC, labels=labels, showmeans=True, meanline=True)
        plt.rcParams['figure.figsize'] = (3, 5)
        plt.boxplot(scores, labels=name_list, showmeans=True, widths=0.4)
        plt.subplots_adjust(left=0.28, right=0.99, top=0.95, bottom=0.05)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.xlabel(r'$\beta$', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.ylim((80, 100))

        # plt.grid(axis="y")
        # plt.show()
        out_dir = os.path.join(data_dir, 'figures')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.sep.join([out_dir, 'overall_performance_' + metric + '_boxplot.tiff']), dpi=300)
        plt.clf()






