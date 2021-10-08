import torch
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


def ramp_up(epoch, max_epoch, max_val, mult):
    if epoch == 0:
        return 0.
    if epoch >= max_epoch:
        return max_val
    return max_val * np.exp(-mult * (1. - float(epoch) / max_epoch) ** 2)


def weight_schedule(epoch, max_epoch, max_val, mult, n_labeled, n_samples):
    max_val = max_val * (float(n_labeled) / n_samples)
    return ramp_up(epoch, max_epoch, max_val, mult)


# Plot Training Diagram
def plot_diagrams(source_accuracy, labeled_target_accuracy, unlabeled_target_accuracy, classification_loss_list_sep,
                  classification_loss_list, general_purpose_loss_list, semantic_transfer_loss_list,
                  prediction_entropy_loss_list, domain_alignment_loss_list, learning_rate_list, duration_list, args,
                  configuration):
    c_list_sep, c_list, g_list, s_list, e_list, d_list = [], [], [], [], [], []
    for i in range(1, args.nepoch + 1):
        if args.separate_classification:
            c_list_sep.append(torch.sum(torch.stack(classification_loss_list_sep[i])).item())
        else:
            c_list_sep.append(0)
        if args.joint_classification:
            c_list.append(torch.sum(torch.stack(classification_loss_list[i])).item())
        else:
            c_list.append(0)
        g_list.append(torch.sum(torch.stack(general_purpose_loss_list[i])).item())
        if args.semantic_loss_weight != 0:
            s_list.append(torch.sum(torch.stack(semantic_transfer_loss_list[i])).item())
        else:
            s_list.append(0)
        if args.domain_loss_weight != 0:
            e_list.append(torch.sum(torch.stack(prediction_entropy_loss_list[i])).item())
        else:
            e_list.append(0)
        if args.domain_loss_weight != 0:
            d_list.append(torch.sum(torch.stack(domain_alignment_loss_list[i])).item())
        else:
            d_list.append(0)

    x = range(1, args.nepoch + 1, 1)
    plt.figure(figsize=(10, 51), dpi=300)

    plt.subplot(8, 1, 1)
    plt.plot(x, source_accuracy, label='SRC', color='deepskyblue')
    plt.fill_between(x, 0, source_accuracy, color='lightskyblue', alpha=0.6)
    plt.plot(x, labeled_target_accuracy, label='LTAR', color='blueviolet')
    plt.fill_between(x, 0, labeled_target_accuracy, color='blueviolet', alpha=0.1)
    plt.plot(x, unlabeled_target_accuracy, label='UTAR', color='green')
    plt.fill_between(x, 0, unlabeled_target_accuracy, color='greenyellow', alpha=0.6)
    target_max_epoch = unlabeled_target_accuracy.index(max(unlabeled_target_accuracy))
    plt.scatter(target_max_epoch, unlabeled_target_accuracy[target_max_epoch], label='MAX', color='red')
    plt.annotate(
        '(' + str(target_max_epoch) + ', ' + '{:.4f}%'.format(unlabeled_target_accuracy[target_max_epoch]) + ')',
        xy=(target_max_epoch, unlabeled_target_accuracy[target_max_epoch]),
        xytext=(target_max_epoch + 0.1, unlabeled_target_accuracy[target_max_epoch] + 0.1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.grid(axis='y')
    plt.legend(bbox_to_anchor=(1, 1), loc=3, borderaxespad=0.5)
    plt.title(
        'Source and Target Accuracy vs Epoch: Task = ' + args.task + ', SRC = ' + args.source + ', TAR = ' + args.target + '\n'
        + 'classification_loss_weight = ' + str(args.classification_loss_weight)
        + ', general_loss_weight = ' + str(args.general_loss_weight)
        + ', semantic_loss_weight = ' + str(args.semantic_loss_weight)
        + ', domain_loss_weight = ' + str(args.domain_loss_weight)
        + ', domain_loss_weight = ' + str(args.domain_loss_weight)
        + ', layer = ' + str(args.layer)
        + ', d_common = ' + str(args.d_common) + '\n'
        + 'lr = ' + str(args.lr)
        # + ', dyn_lr = ' + str(args.dynamic_lr)
        + ', batchsize = ' + str(args.batchsize)
        + ', nepoch = ' + str(args.nepoch) + '\n'
        + 'tau_st = ' + str(args.tau_st)
        + ', tau_tt = ' + str(args.tau_tt) + '\n'
        + 'classifier_turn = ' + str(args.classifier_turn)
        + ', proto_turn (alignment) = ' + str(args.protonet_turn)
        + ', dataset_index = ' + str(configuration['dataset_index']) + '\n'
        + 'ltar_cls_weight = ' + str(args.ltar_cls_weight)
        + ', semantic_transfer_weights = ' + str(args.semantic_transfer_weights) + '\n'
        + 'separate_cls = ' + str(args.separate_classification)
        + ', multiple_iter cls = ' + str(args.multiple_iteration)
        + ', joint_cls = ' + str(args.joint_classification) + '\n'
        + 'target best acc = ' + '{:.4f}%'.format(unlabeled_target_accuracy[target_max_epoch])
        + ', ' + args.time_string)

    plt.subplot(8, 1, 2)
    plt.plot(x, c_list_sep, color='deeppink')
    plt.plot(x, c_list, color='red')
    plt.grid(axis='x')
    plt.fill_between(x, 0, c_list_sep, color='deeppink', alpha=0.3)
    plt.fill_between(x, 0, c_list, color='red', alpha=0.1)
    plt.xlabel('Epoch')
    plt.ylabel('Classification Loss (RED=separate cls loss | PINK=joint cls loss)')
    plt.title('Classification Loss vs Epoch (RED=separate cls loss | PINK=joint cls loss)')

    plt.subplot(8, 1, 3)
    plt.plot(x, g_list, color='orange')
    plt.grid(axis='x')
    plt.fill_between(x, 0, g_list, color='orange', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('General Purpose Loss')
    plt.title('General Purpose Loss vs Epoch')

    plt.subplot(8, 1, 4)
    plt.plot(x, s_list, color='tomato')
    plt.grid(axis='x')
    plt.fill_between(x, 0, s_list, color='tomato', alpha=0.45)
    plt.xlabel('Epoch')
    plt.ylabel('Semantic Transfer Loss')
    plt.title('Semantic Transfer Loss vs Epoch')

    plt.subplot(8, 1, 5)
    plt.plot(x, e_list, color='springgreen')
    plt.grid()
    plt.fill_between(x, 0, e_list, color='springgreen', alpha=0.45)
    plt.xlabel('Epoch')
    plt.ylabel('Prediction Entropy Loss')
    plt.title('Prediction Entropy Loss vs Epoch')

    plt.subplot(8, 1, 6)
    plt.plot(x, d_list, color='aqua')
    plt.grid()
    plt.fill_between(x, 0, d_list, color='aqua', alpha=0.45)
    plt.xlabel('Epoch')
    plt.ylabel('Domain Alignment Loss')
    plt.title('Domain Alignment Loss vs Epoch')

    plt.subplot(8, 1, 7)
    plt.plot(x, learning_rate_list, color='violet')
    plt.grid()
    plt.fill_between(x, 0, learning_rate_list, color='violet', alpha=0.45)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate vs Epoch')

    plt.subplot(8, 1, 8)
    plt.plot(x, duration_list, color='gold')
    plt.plot(x, [np.mean(duration_list) for _ in range(len(x))], color='lightcoral')
    plt.grid()
    plt.fill_between(x, 0, duration_list, color='gold', alpha=0.45)
    plt.xlabel('Epoch')
    plt.ylabel('Epoch Duration')
    plt.title('Duration per Epoch')

    plt.savefig(args.diagram_path)


# Write Log Record
def write_log_record(args, configuration, best_acc):
    with open(args.log_path, 'a') as fp:
        fp.write('PN_HDA: '
                 + '| src = ' + args.source.ljust(4)
                 + '| tar = ' + args.target.ljust(4)
                 + '| best tar acc = ' + str('%.4f' % best_acc).ljust(4)
                 + '| nepoch = ' + str(args.nepoch).ljust(4)
                 + '| layer =' + str(args.layer).ljust(4)
                 + '| d_common =' + str(args.d_common).ljust(4)
                 + '| optimizer =' + str(args.optimizer).ljust(4)
                 + '| lr = ' + str(args.lr).ljust(4)
                 + '| kl_loss =' + str(args.kl_loss).ljust(4)
                 + '| cluster_loss = ' + str(args.cluster_loss).ljust(4)
                 + '| cluster iter =' + str(args.cluster_iter).ljust(4)
                 + '| time = ' + args.time_string
                 # + '| checkpoint_path = ' + str(args.checkpoint_path)
                 + '\n')
    fp.close()


# Command Line Argument Bool Helper
def bool_string(input_string):
    if input_string.lower() not in ['true', 'false']:
        raise ValueError('Bool String Input Invalid! ')
    return input_string.lower() == 'true'


# Maintain a record of the highest accuracy of each task and data partition,
# and the corresponding parameters configuration
def record_maintainer(args, configuration, best_accuracy):
    args.dataset_index = configuration['dataset_index']
    path = os.path.join(args.checkpoint_path, 'logs', 'maintainer.txt')
    if os.path.exists(path):
        maintainer = pickle.load(open(path, 'rb'))
    else:
        maintainer = defaultdict()
    task = ' '.join([args.task.lower(), args.source.lower(), args.target.lower(), str(args.dataset_index)])
    if task in maintainer:
        if maintainer[task][0] < best_accuracy:
            maintainer[task] = [best_accuracy, args]
    else:
        maintainer[task] = [best_accuracy, args]
    pickle.dump(maintainer, open(path, 'wb'))


#  make dirs for model_path, result_path, log_path, diagram_path
def make_dirs(args):
    save_name = '_'.join([args.source.lower(), args.target.lower()])
    log_path = os.path.join(args.checkpoint_path, 'logs')
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(log_path)
        print('Makedir: ' + str(log_path))
    args.log_path = os.path.join(log_path, save_name + '.txt')
    args.avg_path = os.path.join(log_path, save_name + '_avg.txt')
