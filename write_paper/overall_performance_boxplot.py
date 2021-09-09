import numpy as np
import os
import matplotlib.pyplot as plt

class_name = ['background', # 0
                  'L1', # 1
                  'L2', # 2
                  'L3', # 3
                  'L4', # 4
                  'L5', # 5
                  'L1/L2', # 6
                  'L2/L3', # 7
                  'L3/L4', # 8
                  'L4/L5', # 9
                  'L5/S1'] # 10

if __name__ == '__main__':
    data_dir = '/home/pangshumao/data/Spine_Localization_PIL'
    identifier = 'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection-mv'
    data = np.load(os.path.join(data_dir, 'result', identifier, 'all_evaluate_metrics.npz'))

    metrics = 'recall'
    if metrics == 'dice':
        all_dice = data['all_dice']
        title = 'DSC (%)'
    elif metrics == 'precision':
        all_dice = data['all_precision']
        title = 'Precision (%)'
    elif metrics == 'recall':
        all_dice = data['all_recall']
        title = 'Recall (%)'

    all_dice = np.where(all_dice == -1.0, np.nan, all_dice)
    class_num = len(class_name)
    dices = []
    for i in range(1, class_num):
        dice = all_dice[:, i]
        dice = dice[~np.isnan(dice)] * 100

        dices.append(dice)
    labels = class_name[1:]
    # plt.boxplot(DSC, labels=labels, showmeans=True, meanline=True)
    plt.rcParams['figure.figsize'] = (6, 6)
    plt.boxplot(dices, labels=labels, showmeans=True, widths=0.4)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.05)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    # plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel(title, fontsize=14)
    # plt.ylim((81.5, 100))

    # plt.grid(axis="y")
    plt.show()
    out_dir = os.path.join(data_dir, 'figures')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # plt.savefig(os.sep.join([out_dir, 'overall_performance_' + metrics + '.tiff']), dpi=600)






