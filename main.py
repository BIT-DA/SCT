import os
os.environ['MKL_NUM_THREADS']='1'
import time
import datetime
import argparse
import warnings
import torch
import torch.optim as optim
import torch.utils.data
import data_loader_sct

from models_sct import Prototypical
from loss_sct import simpleshot, get_prototype_label_source_2, cal_mean_loss, cal_angle_loss, cal_git_loss, \
    classification_loss_function, compute_cluster_center, cal_cos_loss, make_dirs

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='caltech', help='Source domain',
                    choices=['amazon_surf', 'amazon_decaf', 'amazon_resnet', 'webcam_surf', 'webcam_decaf',
                             'webcam_resnet', 'caltech_surf', 'caltech_decaf', 'caltech_resnet', 'dslr_surf',
                             'dslr_decaf', 'dslr_resnet', 'nustag', 'imagenet',
                             'english', 'french', 'german', 'italian'])
parser.add_argument('--target', type=str, default='amazon', help='Target domain',
                    choices=['amazon_surf', 'amazon_decaf', 'amazon_resnet', 'webcam_surf', 'webcam_decaf',
                             'webcam_resnet', 'caltech_surf', 'caltech_decaf', 'caltech_resnet', 'dslr_surf',
                             'dslr_decaf', 'dslr_resnet', 'nustag', 'imagenet',
                             'spanish5', 'spanish10', 'spanish15', 'spanish20'])
parser.add_argument('--cuda', type=str, default='5', help='Cuda index number')
parser.add_argument('--nepoch', type=int, default=5000, help='Epoch amount')
parser.add_argument('--nepoch_first', type=int, default=1000, help='Epoch amount')
parser.add_argument('--partition', type=int, default=10, help='Number of partition')
parser.add_argument('--layer', type=str, default='double', choices=['single', 'double'],  #
                    help='Structure of the projector network, single layer or double layers projector')
parser.add_argument('--d_common', type=int, default=256, help='Dimension of the common representation')
parser.add_argument('--optimizer', type=str, default='mSGD', choices=['SGD', 'mSGD', 'Adam'], help='optimizer options')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--lr_first', type=float, default=0.01, help='Learning rate')
parser.add_argument('--kd_loss', type=float, default=1.0,
                    help='Trade-off parameter in front of L_kd(lt), set to 0 to turn it off'
                         'Weight the (alpha) * classification loss and (1-alpha) * soft classification loss')
parser.add_argument('--ce_temperature', type=float, default=1.0, help='labeled data temperature CrossEntropy Loss')
parser.add_argument('--kl_loss', type=float, default=0.0,
                    help='Trade-off parameter of L_clu(s, t), set to 0 to turn off, here is kl loss')
parser.add_argument('--cluster_loss', type=float, default=0.0,
                    help='Trade-off parameter of L_clu(s, t), set to 0 to turn off, here is cluster loss')
parser.add_argument('--mean_loss', type=float, default=1.0,
                    help='mean of source and target loss')
parser.add_argument('--angle_loss', type=float, default=1.0,
                    help='mean of source and target loss')
parser.add_argument('--shift_loss', type=float, default=1.0,
                    help='mean of source and target loss')
parser.add_argument('--cos_loss', type=float, default=1.0,
                    help='mean of source and target loss')
parser.add_argument('--git_loss', type=float, default=1.0,
                    help='mean of source and target loss')
parser.add_argument('--git_loss_u', type=float, default=0,
                    help='mean of source and target loss')
parser.add_argument('--margin_loss', type=float, default=0,
                    help='mean of source and target loss')
# parser.add_argument('--cluster_unlabeled', type=bool_string, default=True, #
#                     help='if not use it turns out keep labeled data with more compact intra-class')
parser.add_argument('--combine_pred', type=str, default='Cosine_threshold',  #
                    choices=['Euclidean', 'Cosine', 'Euclidean_threshold', 'Cosine_threshold', 'None'],
                    help='the way of prototype predictions Euclidean(TPN), Cosine(PFAN), None(not use)')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint', help='All records save path')
parser.add_argument('--cos_threshold', type=float, default=0,
                    help='how much margin is in cos_threshold ')
parser.add_argument('--cluster_method', type=str, default='kernel_kmeans',
                    help='cluster method, choice=[spherical_k_means, k_means, kernel_kmeans]')
parser.add_argument('--cluster_iter', type=int, default=5, help='cluster iteration')
parser.add_argument('--pre_lr', type=str, default='0', help='learning rate of pretrain best model')
args = parser.parse_args()
args.time_string = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if len(args.cuda) == 1:
        torch.cuda.set_device(int(args.cuda))
make_dirs(args)
print(str(args))


def tst(model, configuration, srctar):
    model.eval()
    if srctar == 'source':
        loader = configuration['source_data']
        s_loader = configuration['source_data']
        l_loader = configuration['labeled_target_data']
        N = configuration['ns']
    elif srctar == 'labeled_target':
        loader = configuration['labeled_target_data']
        s_loader = configuration['source_data']
        l_loader = configuration['labeled_target_data']
        N = configuration['nl']
    elif srctar == 'unlabeled_target':
        loader = configuration['unlabeled_target_data']
        s_loader = configuration['source_data']
        l_loader = configuration['labeled_target_data']
        N = configuration['nu']
    else:
        raise Exception('Parameter srctar invalid! ')

    with torch.no_grad():
        feature, label = loader[0].float(), loader[1].reshape(-1, ).long()
        feature_s, label_s = s_loader[0].float(), s_loader[1].reshape(-1, ).long()
        feature_l, label_l = l_loader[0].float(), l_loader[1].reshape(-1, ).long()
        if torch.cuda.is_available():
            feature, label = feature.cuda(), label.cuda()
            feature_s, label_s = feature_s.cuda(), label_s.cuda()
            feature_l, label_l = feature_l.cuda(), label_l.cuda()
        # classifier_output, _ = model(input_feature=feature)
        learned_feature = model(input_feature=feature)
        source_learned_feature = model(input_feature=feature_s)
        l_target_learned_feature = model(input_feature=feature_l)

        s_mean = torch.mean(source_learned_feature, dim=0)
        source_learned_feature -= s_mean
        learned_feature -= s_mean

        prototype_output, _, _ = get_prototype_label_source_2(source_learned_features=source_learned_feature,
                                                              u_target_learned_features=learned_feature,
                                                              source_labels=label_s,
                                                              configuration=configuration,
                                                              combine_pred=args.combine_pred,
                                                              epoch=0)
        _, pred = torch.max(prototype_output.data, 1)
        n_correct = (pred == label).sum().item()
        acc = float(n_correct) / N * 100.

    return acc


def train(model, optimizer, configuration):
    best_acc = -float('inf')

    # For diagram drawing
    duration_list, source_accuracy, labeled_target_accuracy, unlabeled_target_accuracy = [], [], [], []

    # training
    # kmeans->聚类
    # 类对齐
    for epoch in range(args.nepoch):
        # scheduler.step()
        if epoch == args.nepoch_first:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        if epoch % 500 == 0:
            print('optimizer.state_dict', optimizer.state_dict()['param_groups'][0]['lr'])
        # print('当前的epoch:', epoch)
        # print('============================%d epoch=======================' % epoch)
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        # prepare data
        source_data = configuration['source_data']
        l_target_data = configuration['labeled_target_data']
        u_target_data = configuration['unlabeled_target_data']
        source_feature, source_label = source_data[0].float(), source_data[1].reshape(-1, ).long()
        l_target_feature, l_target_label = l_target_data[0].float(), l_target_data[1].reshape(-1, ).long()
        u_target_feature, u_target_label = u_target_data[0].float(), u_target_data[1].reshape(-1, ).long()
        if torch.cuda.is_available():
            source_feature, source_label = source_feature.cuda(), source_label.cuda()
            l_target_feature, l_target_label = l_target_feature.cuda(), l_target_label.cuda()
            u_target_feature = u_target_feature.cuda()
        # 初始pretrain模型测试
        if epoch == 0:
            acc_src = tst(model, configuration, 'source')
            acc_labeled_tar = tst(model, configuration, 'labeled_target')
            acc_unlabeled_tar = tst(model, configuration, 'unlabeled_target')
            model.train()
            end_time = time.time()
            print('ACC -> ', end='')
            print('Epoch: [{}/{}], {:.1f}s, Src acc: {:.4f}%, LTar acc: {:.4f}%, UTar acc: {:.4f}%'.format(
                epoch, args.nepoch, end_time - start_time, acc_src, acc_labeled_tar, acc_unlabeled_tar))

        source_learned_feature = model(input_feature=source_feature)
        l_target_learned_feature = model(input_feature=l_target_feature)
        u_target_learned_feature = model(input_feature=u_target_feature)
        sub = torch.mean(source_learned_feature, dim=0)

        # simple shot
        source_learned_feature = simpleshot(source_learned_feature, sub)
        l_target_learned_feature = simpleshot(l_target_learned_feature, sub)
        u_target_learned_feature = simpleshot(u_target_learned_feature, sub)

        error_overall = 0.0

        # 根据 source 获取伪标签
        u_target_output, u_target_pseudo_label, u_target_pseudo_label_all = get_prototype_label_source_2(
            source_learned_features=source_learned_feature,
            u_target_learned_features=u_target_learned_feature,
            source_labels=source_label,
            configuration=configuration,
            combine_pred=args.combine_pred,
            epoch=epoch)


        target_learned_feature_all = torch.cat((l_target_learned_feature, u_target_learned_feature), dim=0)
        # target_label_all = torch.cat((l_target_label.cpu(), u_target_label), dim=0)
        target_pseudo_label = torch.cat((l_target_label, u_target_pseudo_label_all))

        # target 类中心计算
        target_refine_pseudo_label = target_pseudo_label
        target_cluster_center = compute_cluster_center(features=target_learned_feature_all,
                                                       labels=target_refine_pseudo_label,
                                                       class_num=configuration['class_number'])

        # source 类中心计算
        source_cluster_center = compute_cluster_center(features=source_learned_feature,
                                                       labels=source_label,
                                                       class_num=configuration['class_number'])

        source_one_center = torch.mean(source_learned_feature, dim=0)
        target_one_center = torch.mean(target_learned_feature_all, dim=0)
        # 计算全局中心loss
        if args.mean_loss and epoch >= args.nepoch_first:
            mean_loss, source_one_center, target_one_center = cal_mean_loss(source_learned_feature,
                                                                            target_learned_feature_all)
            error_overall += mean_loss * args.mean_loss
            if epoch % 10 == 0:
                print('Use mean_loss: ', mean_loss * args.mean_loss)

        # 计算source CrossEntropyLoss loss
        if args.combine_pred.find('Euclidean') != -1 or args.combine_pred.find('Cosine') != -1:
            source_output, _, _ = get_prototype_label_source_2(source_learned_features=source_learned_feature,
                                                               u_target_learned_features=source_learned_feature,
                                                               source_labels=source_label,
                                                               configuration=configuration,
                                                               combine_pred=args.combine_pred,
                                                               epoch=epoch)
            classification_loss = classification_loss_function(source_output, source_label, args.ce_temperature)
            error_overall += classification_loss
            if epoch % 10 == 0:
                print('Use source CE loss: ', classification_loss)

        # 计算target CrossEntropyLoss loss
        if args.combine_pred.find('Euclidean') != -1 or args.combine_pred.find('Cosine') != -1:
            l_target_output, _, _ = get_prototype_label_source_2(source_learned_features=source_learned_feature,
                                                                 u_target_learned_features=l_target_learned_feature,
                                                                 source_labels=source_label,
                                                                 configuration=configuration,
                                                                 combine_pred=args.combine_pred,
                                                                 epoch=epoch)
            l_classification_loss = classification_loss_function(l_target_output, l_target_label, args.ce_temperature)
            error_overall += l_classification_loss
            if epoch % 10 == 0:
                print('Use l target loss: ', l_classification_loss)

        # 计算每个类中心欧式距离的loss 和 角度夹角之间的loss
        if (args.cos_loss or args.shift_loss) and (epoch >= args.nepoch_first or not pre_lr_str.strip() == '0'):
            shift_loss, cos_loss = cal_cos_loss(source_one_center, target_one_center,
                                                   source_cluster_center, target_cluster_center)
            if args.shift_loss:
                error_overall += shift_loss * args.shift_loss
            if args.cos_loss:
                error_overall += cos_loss * args.cos_loss
            if epoch % 10 == 0 and args.cos_loss:
                print('Use cos_loss:', cos_loss * args.cos_loss)
            if epoch % 10 == 0 and args.shift_loss:
                print('Use shift_loss:', shift_loss * args.shift_loss)

        # backward propagation
        error_overall.backward()
        optimizer.step()

        # tsting Phase
        acc_src = tst(model, configuration, 'source')
        acc_labeled_tar = tst(model, configuration, 'labeled_target')
        acc_unlabeled_tar = tst(model, configuration, 'unlabeled_target')
        end_time = time.time()
        print('ACC -> ', end='')
        print('Epoch: [{}/{}], {:.1f}s, Src acc: {:.4f}%, LTar acc: {:.4f}%, UTar acc: {:.4f}%'.format(
            epoch, args.nepoch, end_time - start_time, acc_src, acc_labeled_tar, acc_unlabeled_tar))
        # For diagram drawing
        source_accuracy.append(acc_src)
        labeled_target_accuracy.append(acc_labeled_tar)
        unlabeled_target_accuracy.append(acc_unlabeled_tar)
        duration_list.append(end_time - start_time)

        if best_acc < acc_unlabeled_tar:
            best_acc = acc_unlabeled_tar
            best_text = args.source.ljust(10) + '-> ' + args.target.ljust(
                10) + ' The proposed model for HDA achieves current best accuracy. '
            print(best_text)
            if epoch >= 1000:
                print('need more epoch training')

    # end for max_epoch
    print('Best tst Accuracy: {:.4f}%'.format(best_acc))
    # write_log_record(args, configuration, best_acc)
    return best_acc


if __name__ == '__main__':
    result = 0.
    for i in range(args.partition):
        print(args)
        print(os.path.basename(__file__))
        pre_lr_str = args.pre_lr
        # if not pre_lr_str.strip() == '0':
        #     configuration = data_loader.get_configuration(args, is_train=True)
        # else:
        #     configuration = data_loader.get_configuration(args, is_train=False)
        configuration = data_loader_sct.get_configuration(args)

        model = Prototypical(configuration['d_source'], configuration['d_target'], args.d_common,
                             configuration['class_number'], args.layer)
        if torch.cuda.is_available():
            model = model.cuda()
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'mSGD':
            optimizer = optim.SGD(model.parameters(), lr=args.lr_first, momentum=0.9,
                                  weight_decay=0.001, nesterov=True)
            gamm = '%.4f'%(args.lr / args.lr_first)
            print('lr gam', gamm)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000], gamma=gamm, last_epoch=-1)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

        # a_surf->a_decaf ./save/300-epoch.pth
        if not pre_lr_str.strip() == '0':
            checkpoint = torch.load(
                './ppa_save_spt_{}_{}/{}_best-epoch.pth'.format(args.source, args.target, args.pre_lr),
                map_location='cuda:%s' % args.cuda)
            model.load_state_dict(checkpoint['feature_extractor'])
            print('load model............', args.source, args.target, args.pre_lr)
        result += train(model, optimizer, configuration)
    print('Avg acc:', str('%.4f' % (result / args.partition)).ljust(4))
