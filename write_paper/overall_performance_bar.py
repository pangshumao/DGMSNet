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
    metrics = ['DSC', 'Precision', 'Recall']
    identifiers = [
        'segmentation_DeepLabv3_plus_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_200',
        'segmentation_DeepLabv3_plus_gcn_Adam_lr_0.005_CrossEntropyLoss_batch_size_4_epochs_50_beta_0.3_gcn_num_2_pre_trained',
        'MRLN_Adam_lr_0.005_epoch_400_batch_size_4_beta_40.0',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_30.0_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_beta_50.0_use_weak_label_batch_size_4_epochs_200',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_majority-voting',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection',
        'multi-task_DeepLabv3_plus_lil_Adam_lr_0.005_use_weak_label_batch_size_4_epochs_200_ensemble_type_model-selection-mv']

    name_list = ['Vertebrae', 'IVDs', 'All']

    label_list = ['DGSSNet-w/o-D',
                  'GCSN',
                  'MRLN',
                  'DGSSNet-w/o-W',
                  'DGSSNet-w/o-DGLF',
                  'DGSSNet-w/-MV',
                  'DGSSNet-w/-AMS',
                  'DGSSNet']
    for metric in metrics:
        all_ver_mean_dices = []
        all_ivd_mean_dices = []
        all_mean_dices = []

        all_ver_std_dices = []
        all_ivd_std_dices = []
        all_std_dices = []

        for identifier in identifiers:
            data = np.load(os.path.join(data_dir, 'result', identifier, 'all_evaluate_metrics.npz'))
            if metric == 'DSC':
                subject_ver_dices = data['subject_ver_dices']
                subject_ivd_dices = data['subject_ivd_dices']
                subject_dices = data['subject_dices']
            elif metric == 'Precision':
                subject_ver_dices = data['subject_ver_precisions']
                subject_ivd_dices = data['subject_ivd_precisions']
                subject_dices = data['subject_precisions']
            elif metric == 'Recall':
                subject_ver_dices = data['subject_ver_recalls']
                subject_ivd_dices = data['subject_ivd_recalls']
                subject_dices = data['subject_recalls']

            all_ver_mean_dices.append(np.mean(subject_ver_dices) * 100)
            all_ver_std_dices.append(np.std(subject_ver_dices) * 100)

            all_ivd_mean_dices.append(np.mean(subject_ivd_dices) * 100)
            all_ivd_std_dices.append(np.std(subject_ivd_dices) * 100)

            all_mean_dices.append(np.mean(subject_dices) * 100)
            all_std_dices.append(np.std(subject_dices) * 100)

        x = list(range(len(name_list)))
        total_width = 0.8
        n = len(identifiers)
        width = total_width / n

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 7)  # for dpi=300, output = w * h pixels

        plt.gca().xaxis.set_major_locator(plt.NullLocator())

        for i in range(0, n):
            if i > 0:
                for j in range(len(x)):
                    x[j] = x[j] + width

            if i == (n // 2 - 1):
                plt.bar(x, height=[all_ver_mean_dices[i], all_ivd_mean_dices[i], all_mean_dices[i]],
                        width=width,
                        yerr=[all_ver_std_dices[i], all_ivd_std_dices[i], all_std_dices[i]],
                        label=label_list[i], tick_label=name_list)
            else:
                plt.bar(x, height=[all_ver_mean_dices[i], all_ivd_mean_dices[i], all_mean_dices[i]],
                        width=width,
                        yerr=[all_ver_std_dices[i], all_ivd_std_dices[i], all_std_dices[i]],
                        label=label_list[i])
            # plt.text(x[0] - width, all_ver_mean_dices[i] + 0.1, "%.2f" % all_ver_mean_dices[i], fontsize=10)
            # plt.text(x[1] - width, all_ivd_mean_dices[i] + 0.1, "%.2f" % all_ivd_mean_dices[i], fontsize=10)
            # plt.text(x[2] - width, all_mean_dices[i] + 0.1, "%.2f" % all_mean_dices[i], fontsize=10)

        plt.ylim((40, 102))
        plt.yticks(list(range(40, 102, 5)), fontsize=14)
        plt.xticks(fontsize=14)
        plt.grid(axis='y', color=(0.8, 0.8, 0.8), alpha=0.2)
        plt.subplots_adjust(left=0.15, right=0.99, top=0.95, bottom=0.05)
        # plt.yticks(fontsize=12)
        plt.ylabel(metric + ' (%)', fontsize=14)
        # plt.legend(fontsize=14, loc='upper right')
        plt.legend(fontsize=12, loc='lower right')
        plt.show()
        out_dir = os.path.join(data_dir, 'figures')
        # plt.savefig(os.sep.join([out_dir, 'overall_performance_' + metric + '_bar.tiff']), dpi=300)


