import torch
from scipy.sparse import csr_matrix
import gc
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kernel_kmeans import KernelKMeans
from collections import OrderedDict
from scipy.spatial.distance import cdist


def classification_loss_function(prediction, true_labels, ce_temperature=1.0):
    celoss_criterion = nn.CrossEntropyLoss()
    return celoss_criterion(prediction / ce_temperature, true_labels)


def simpleshot(feature, sub):
    sub_feature = feature - sub
    sub_feature = sub_feature / sub_feature.norm(dim=1, p=2, keepdim=True)
    return sub_feature

def select_pred(pred, threshold=0.0):
    top_k, indices = torch.topk(pred, 2)
    num = pred.shape[0]
    select_list = []
    for index in range(num):
        dec = top_k[index, 0] - top_k[index, 1]
        if dec > threshold:
            select_list.append(index)
            # print(index)
    return select_list
    # bb = pred.index_select(dim=0, index=torch.tensor(select_list))
    # print(bb)


def get_prototype_label_source_2(source_learned_features, u_target_learned_features, source_labels, configuration,
                                 combine_pred, epoch):
    """
    get unlabeled target prototype label
    :param epoch: training epoch
    :param combine_pred: Euclidean, Cosine
    :param configuration: dataset configuration
    :param source_learned_features: source feature
    :param l_target_learned_features:  labeled target feature
    :param u_target_learned_features:  unlabeled target feature
    :param source_labels: source labels
    :param l_target_labels: labeled target labels
    :return: unlabeled target prototype label
    """

    def prototype_softmax(features, feature_centers):
        assert features.shape[1] == feature_centers.shape[1]
        n_samples = features.shape[0]
        pred = torch.FloatTensor()
        C, dim = feature_centers.shape
        for i in range(n_samples):
            if combine_pred.find('Euclidean') != -1:
                dis = -torch.sum(torch.pow(features[i].expand(C, dim) - feature_centers, 2), dim=1)
            elif combine_pred.find('Cosine') != -1:
                dis = torch.cosine_similarity(features[i].expand(C, dim), feature_centers)
            if not i:
                pred = dis.reshape(1, -1)  # 1xC
            else:
                pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)
        return pred  # NXC

    assert source_learned_features.shape[1] == u_target_learned_features.shape[1]
    class_num = configuration['class_number']
    feature_dim = source_learned_features.shape[1]
    feature_centers = torch.zeros((class_num, feature_dim))
    for k in range(class_num):
        # calculate feature center of each class for source and target
        k_source_feature = source_learned_features.index_select(dim=0,
                                                                index=(source_labels == k).nonzero().reshape(-1, ))
        # k_l_target_feature = l_target_learned_features.index_select(dim=0, index=(
        #         l_target_labels == k).nonzero().reshape(-1, ))
        # feature_centers[k] = torch.mean(torch.cat((k_source_feature, k_l_target_feature), dim=0), dim=0)
        feature_centers[k] = torch.mean(k_source_feature, dim=0)

    if torch.cuda.is_available():
        feature_centers = feature_centers.cuda()

    # assign 'pseudo label' by Euclidean distance or Cosine similarity between feature and prototype,
    # select the most confident samples in each pseudo class, not confident label=-1
    prototype_pred = prototype_softmax(u_target_learned_features, feature_centers)
    prototype_value, prototype_label = torch.max(prototype_pred.data, 1)
    select_label = prototype_label.detach().clone()
    # add threshold
    if combine_pred.find('threshold') != -1:
        if combine_pred == 'Euclidean_threshold':
            # threshold for Euclidean distance
            # select_threshold = 0.2
            select_threshold = 0.6
        elif combine_pred == 'Cosine_threshold':
            # Ref: Progressive Feature Alignment for Unsupervised Domain Adaptation CVPR2019
            # select_threshold = 1. / (1 + np.exp(-0.8 * (epoch + 1))) - 0.01
            # select_threshold = 0.1
            select_threshold = 0.9
            select_label[(prototype_value < select_threshold).nonzero()] = -1

    return prototype_pred, select_label, prototype_label


def cal_one_loss(source_cluster_center, l_target_learned_feature, l_target_label):
    C, dim = source_cluster_center.shape
    feature_dim = l_target_learned_feature.size(1)
    feature_centers = torch.zeros((C, feature_dim)).cuda()
    n_samples = l_target_learned_feature.shape[0]
    pred = torch.FloatTensor()

    loss = 0.0
    for i in range(n_samples):
        dis = torch.cosine_similarity(l_target_learned_feature[i].expand(C, dim), source_cluster_center)
        up = dis[l_target_label[i]]
        down = C - 1 - (torch.sum(dis) - up)
        loss += (1 - up) / down
    loss /= n_samples
    return loss


def multi_proto_softmax(features, s_centers, l_centers, combine_pred='Cosine'):
    """
    根据source中心和l_target中心共同计算 距离
    :param features:
    :param s_centers:
    :param l_centers:
    :param combine_pred:
    :return:
    """
    assert features.shape[1] == s_centers.shape[1]
    n_samples = features.shape[0]
    pred_s = torch.FloatTensor()
    pred_l = torch.FloatTensor()
    C, dim = s_centers.shape
    for i in range(n_samples):
        if combine_pred.find('Euclidean') != -1:
            # 加上负号是因为要根据softmax选取最大的值 （-max,0]
            dis_s = -torch.sum(torch.pow(features[i].expand(C, dim) - s_centers, 2), dim=1)
            dis_max_s = torch.max(-dis)
            dis_s = dis_s / dis_max_s

            dis = -torch.sum(torch.pow(features[i].expand(C, dim) - l_centers, 2), dim=1)
            dis_max = torch.max(-dis)
            dis = dis / dis_max
        elif combine_pred.find('Cosine') != -1:
            # 计算cos距离 [-1,1]
            dis_s = torch.cosine_similarity(features[i].expand(C, dim), s_centers)
            dis = torch.cosine_similarity(features[i].expand(C, dim), l_centers)

        if not i:
            pred_s = dis_s.reshape(1, -1)  # 1xC
            pred_l = dis.reshape(1, -1)  # 1xC
        else:
            pred_s = torch.cat((pred_s, dis_s.reshape(1, -1)), dim=0)
            pred_l = torch.cat((pred_l, dis.reshape(1, -1)), dim=0)
    return pred_l + pred_s  # NXC


def get_pseudo_label(u_features, s_centers, l_centers, combine_pred='Cosine'):
    prototype_pred = multi_proto_softmax(u_features, s_centers, l_centers, combine_pred)
    prototype_value, prototype_label = torch.max(prototype_pred.data, 1)
    return prototype_pred, prototype_label


def pred_s_label(u_features, labeled_centers, combine_pred='Cosine'):
    """
    对全部的 unlabeled target 分配伪装标签
    主要用于最后一次的train加入全部unlabeled target 和 测试正确率 的步骤中
    :param u_features:
    :param labeled_centers:
    :param topk:
    :param combine_pred:
    :return:
    """
    n_samples = u_features.shape[0]
    pred = torch.FloatTensor()
    C, dim = labeled_centers.shape
    for i in range(n_samples):
        if combine_pred.find('Euclidean') != -1:
            dis = -torch.sum(torch.pow(u_features[i].expand(C, dim) - labeled_centers, 2), dim=1)
        elif combine_pred.find('Cosine') != -1:
            dis = torch.cosine_similarity(u_features[i].expand(C, dim), labeled_centers)
        if not i:
            pred = dis.reshape(1, -1)  # 1xC
        else:
            pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)

    prototype_value, prototype_label = torch.max(pred.data, 1)
    return prototype_value, prototype_label

def get_pred(u_features, labeled_centers, combine_pred='Cosine'):
    """
    对全部的 unlabeled target 分配伪装标签
    主要用于最后一次的train加入全部unlabeled target 和 测试正确率 的步骤中
    :param u_features:
    :param labeled_centers:
    :param topk:
    :param combine_pred:
    :return:
    """
    n_samples = u_features.shape[0]
    pred = torch.FloatTensor()
    C, dim = labeled_centers.shape
    for i in range(n_samples):
        if combine_pred.find('Euclidean') != -1:
            dis = -torch.sum(torch.pow(u_features[i].expand(C, dim) - labeled_centers, 2), dim=1)
        elif combine_pred.find('Cosine') != -1:
            dis = torch.cosine_similarity(u_features[i].expand(C, dim), labeled_centers)
        if not i:
            pred = dis.reshape(1, -1)  # 1xC
        else:
            pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)
    return pred


def select_source_sample(s_features, s_labels, l_centers, combine_pred='Cosine'):
    s_pred_value, s_pred_labels = pred_s_label(s_features, l_centers, combine_pred)
    # print('s_pred_labels_pred', s_pred_labels.shape)
    # print('s_labels.shape', s_labels.shape)
    select_indx = torch.eq(s_pred_labels, s_labels)
    print(s_pred_value[select_indx].mean())
    return select_indx


def multi_proto_softmax_weight(features, s_centers, l_centers, combine_pred='Cosine', lt_weight=0.5):
    """
    根据source中心和l_target中心共同计算 距离 ； lt给予一定的权重
    :param lt_weight: lt的权重
    :param features:
    :param s_centers:
    :param l_centers:
    :param combine_pred:
    :return:
    """
    # print(lt_weight)
    assert features.shape[1] == s_centers.shape[1]
    n_samples = features.shape[0]
    pred_s = torch.FloatTensor()
    pred_l = torch.FloatTensor()
    C, dim = s_centers.shape
    for i in range(n_samples):
        if combine_pred.find('Euclidean') != -1:
            # 加上负号是因为要根据softmax选取最大的值 （-max,0]
            dis_s = -torch.sum(torch.pow(features[i].expand(C, dim) - s_centers, 2), dim=1)
            dis_max_s = torch.max(-dis)
            dis_s = dis_s / dis_max_s

            dis = -torch.sum(torch.pow(features[i].expand(C, dim) - l_centers, 2), dim=1)
            dis_max = torch.max(-dis)
            dis = dis / dis_max
        elif combine_pred.find('Cosine') != -1:
            # 计算cos距离 [-1,1]
            dis_s = torch.cosine_similarity(features[i].expand(C, dim), s_centers)
            dis = torch.cosine_similarity(features[i].expand(C, dim), l_centers)

        if not i:
            pred_s = dis_s.reshape(1, -1)  # 1xC
            pred_l = dis.reshape(1, -1)  # 1xC
        else:
            pred_s = torch.cat((pred_s, dis_s.reshape(1, -1)), dim=0)
            pred_l = torch.cat((pred_l, dis.reshape(1, -1)), dim=0)
    return pred_l * lt_weight + pred_s  # NXC


def get_pseudo_label_weight(u_features, s_centers, l_centers, combine_pred='Cosine', lt_weight=0.5):
    prototype_pred = multi_proto_softmax_weight(u_features, s_centers, l_centers, combine_pred, lt_weight)
    prototype_value, prototype_label = torch.max(prototype_pred.data, 1)
    return prototype_pred, prototype_label


def get_l2_and_cos_proto_softmax(features, labels_centers):
    assert features.shape[1] == labels_centers.shape[1]
    n_samples = features.shape[0]
    pred = torch.FloatTensor()

    C, dim = labels_centers.shape
    for i in range(n_samples):
        # 加上负号是因为要根据softmax选取最大的值 （-max,0]
        dis = -torch.sum(torch.pow(features[i].expand(C, dim) - labels_centers, 2), dim=1)
        dis_max = torch.max(-dis)
        # 欧式距离
        dis = dis / dis_max
        # cos 距离
        dis_cos = torch.cosine_similarity(features[i].expand(C, dim), labels_centers)

        dis = dis + dis_cos / 2
        # print('dis.shape', dis.shape)
        if not i:
            pred = dis.reshape(1, -1)  # 1xC
        else:
            pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)
    return pred  # NXC


def get_l2_and_cos_pseudo_label(u_features, labeled_centers):
    prototype_pred = get_l2_and_cos_proto_softmax(u_features, labeled_centers)
    prototype_value, prototype_label = torch.max(prototype_pred.data, 1)
    return prototype_pred, prototype_label


def get_select_pseudo_label(u_features, labeled_centers, topk, combine_pred='Cosine'):
    """
    选取 topk 个过softmax之后的 u_label 的 特征和伪标签
    :param u_features:
    :param labeled_centers:
    :param topk:
    :param combine_pred:
    :return: 选取的特征和伪标签
    """
    n_samples = u_features.shape[0]
    pred = torch.FloatTensor()
    C, dim = labeled_centers.shape
    for i in range(n_samples):
        if combine_pred.find('Euclidean') != -1:
            dis = -torch.sum(torch.pow(u_features[i].expand(C, dim) - labeled_centers, 2), dim=1)
        elif combine_pred.find('Cosine') != -1:
            dis = torch.cosine_similarity(u_features[i].expand(C, dim), labeled_centers)
        if not i:
            pred = dis.reshape(1, -1)  # 1xC
        else:
            pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)

    prototype_value, prototype_label = torch.max(pred.data, 1)
    # print('prototype_value', prototype_value)
    # print('prototype_label', prototype_label)
    proto_val = prototype_value.data.cpu().numpy()
    # print(proto_val)
    proto_sort = np.argsort(proto_val)
    # print(proto_sort)
    proto_topk = proto_sort[-topk:]
    # print(proto_topk[:10])

    select_index = torch.from_numpy(proto_topk).cuda()
    # select_feature = u_features.index_select(0, select_index)
    # select_proto_value = prototype_value.index_select(0, select_index)
    select_proto_label = prototype_label.index_select(0, select_index)
    return select_index, select_proto_label


def get_all_u_pseudo_label(u_features, labeled_centers, combine_pred='Cosine'):
    """
    对全部的 unlabeled target 分配伪装标签
    主要用于最后一次的train加入全部unlabeled target 和 测试正确率 的步骤中
    :param u_features:
    :param labeled_centers:
    :param topk:
    :param combine_pred:
    :return:
    """
    n_samples = u_features.shape[0]
    pred = torch.FloatTensor()
    C, dim = labeled_centers.shape
    for i in range(n_samples):
        if combine_pred.find('Euclidean') != -1:
            dis = -torch.sum(torch.pow(u_features[i].expand(C, dim) - labeled_centers, 2), dim=1)
        elif combine_pred.find('Cosine') != -1:
            dis = torch.cosine_similarity(u_features[i].expand(C, dim), labeled_centers)
        if not i:
            pred = dis.reshape(1, -1)  # 1xC
        else:
            pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)

    prototype_value, prototype_label = torch.max(pred.data, 1)
    return prototype_label


def get_pred_pseudo_label(u_features, labeled_centers, combine_pred='Cosine'):
    """
    对全部的 unlabeled target 分配伪装标签
    用于 Source 和 L_Target 的 CrossEntropyLoss()
    :param u_features:
    :param labeled_centers:
    :param topk:
    :param combine_pred:
    :return:
    """
    n_samples = u_features.shape[0]
    pred = torch.FloatTensor()
    C, dim = labeled_centers.shape
    for i in range(n_samples):
        if combine_pred.find('Euclidean') != -1:
            dis = -torch.sum(torch.pow(u_features[i].expand(C, dim) - labeled_centers, 2), dim=1)
        elif combine_pred.find('Cosine') != -1:
            dis = torch.cosine_similarity(u_features[i].expand(C, dim), labeled_centers)
        if not i:
            pred = dis.reshape(1, -1)  # 1xC
        else:
            pred = torch.cat((pred, dis.reshape(1, -1)), dim=0)

    return pred


def cal_mean_loss(source_learned_feature, target_learned_feature):
    """
    计算源域和目标域的全局中心的欧氏距离
    :param source_learned_feature: 源域特征
    :param target_learned_feature: 目标域特征
    :return: 全局中心欧式距离，源域全局中心，目标域全局中心
    """
    source_one_center = torch.mean(source_learned_feature, dim=0)
    target_one_center = torch.mean(target_learned_feature, dim=0)
    dist = F.pairwise_distance(source_one_center.reshape(1, -1), target_one_center.reshape(1, -1), p=2)  # pytorch求欧氏距离
    # print('dist.shape', dist.shape)
    return dist, source_one_center, target_one_center


def cal_angle_loss(source_center, target_center, source_cluster_center, target_cluster_center, shift_weight=1.0):
    shift_source_cluster_center = source_cluster_center - source_center
    shift_target_cluster_center = target_cluster_center - target_center
    len_class = shift_source_cluster_center.size(0)
    distance = torch.zeros(len_class).cuda()
    class_center_distance = torch.zeros(len_class).cuda()
    for i in range(len_class):
        class_center_distance[i] = F.pairwise_distance(shift_source_cluster_center[i].reshape(1, -1),
                                                       shift_target_cluster_center[i].reshape(1, -1),
                                                       p=2).item()
        distance[i] = 1 - torch.cosine_similarity(shift_source_cluster_center[i], shift_target_cluster_center[i], dim=0)
    shift_loss = distance.sum() * shift_weight
    dist_loss = class_center_distance.sum()
    # print(shift_loss.shape)
    # print(dist_loss.shape)
    return shift_loss, dist_loss

def cal_center_sim(source_center, target_center, len_class):
    class_center_distance = torch.zeros(len_class).cuda()
    for i in range(len_class):
        class_center_distance[i] = torch.cosine_similarity(source_center[i], target_center[i], dim=0)

    # for i in range(len_class):
    #     class_center_distance[i] = F.pairwise_distance(source_center[i].reshape(1, -1), target_center[i].reshape(1, -1), p=2)  # pytorch求欧氏距离

    print('----end sim----', class_center_distance.mean())
    return class_center_distance

def cal_class_sim(source_center, target_center, len_class):
    class_center_distance = 0.0
    for i in range(len_class):
        class_center_distance += F.pairwise_distance(source_center[i].reshape(1, -1), target_center[i].reshape(1, -1), p=2)
    return class_center_distance / len_class

def cal_ul_loss(source_center, target_center, len_class):
    class_center_distance = torch.zeros(len_class).cuda()
    for i in range(len_class):
        class_center_distance[i] = 1 - torch.cosine_similarity(source_center[i], target_center[i], dim=0)
    return class_center_distance.mean()


def cal_cos_loss(source_center, target_center, source_cluster_center, target_cluster_center):
    """
    计算源域和目标域上 每个类的中心到其他类的中心的cos距离 之差  class_num * (class_num)/2 个 的平和
    :param source_center: 源域的全局中心
    :param target_center: 目标域的全局中心
    :param source_cluster_center: 源域的各类中心
    :param target_cluster_center: 目标域的各类中心
    :return: 每个类的中心到其他类的中心的cos距离 之差 的和， 每个类减去全局中心的对齐的loss值的和
    """
    # print(source_center)
    shift_source_cluster_center = source_cluster_center - source_center
    shift_target_cluster_center = target_cluster_center - target_center
    # print('shift_target_cluster_center.shape', shift_target_cluster_center.shape)
    len_class = shift_source_cluster_center.size(0)

    distance = 0.0
    for ii in range(len_class):
        for jj in range(ii, len_class):
            sij = torch.cosine_similarity(shift_source_cluster_center[ii], shift_source_cluster_center[jj], dim=0)
            tij = torch.cosine_similarity(shift_target_cluster_center[ii], shift_target_cluster_center[jj], dim=0)
            # print(type(sij))
            distance += torch.abs(sij - tij)
    distance /= (ii * (ii - 1) / 2)

    class_center_distance = torch.zeros(len_class).cuda()
    for i in range(len_class):
        class_center_distance[i] = F.pairwise_distance(source_cluster_center[i].reshape(1, -1),
                                                       target_cluster_center[i].reshape(1, -1),
                                                       p=2).item()
    shift_loss = class_center_distance.sum()
    return shift_loss, distance


def cal_cos_loss_aver(source_center, target_center, source_cluster_center, target_cluster_center,
                      combine_pred='Euclidean'):
    """
    计算源域和目标域上 每个类的中心到其他类的中心的cos距离 之差  class_num * (class_num)/2 个 的平均值
    :param source_center: 源域的全局中心
    :param target_center: 目标域的全局中心
    :param source_cluster_center: 源域的各类中心
    :param target_cluster_center: 目标域的各类中心
    :return: 每个类的中心到其他类的中心的cos距离 之差 的平均值， 每个类减去全局中心的对齐的loss值平均值
    """
    shift_source_cluster_center = source_cluster_center - source_center
    shift_target_cluster_center = target_cluster_center - source_center
    # print('shift_target_cluster_center.shape', shift_target_cluster_center.shape)
    len_class = shift_source_cluster_center.size(0)

    distance = 0.0
    for ii in range(len_class):
        for jj in range(ii, len_class):
            sij = torch.cosine_similarity(shift_source_cluster_center[ii], shift_source_cluster_center[jj], dim=0)
            tij = torch.cosine_similarity(shift_target_cluster_center[ii], shift_target_cluster_center[jj], dim=0)
            # print(type(sij))
            distance += torch.abs(sij - tij)
    distance /= (ii * (ii - 1) / 2)

    class_center_distance = torch.zeros(len_class).cuda()

    if combine_pred.find('Euclidean') != -1:
        print('using Euclidean cal_cos_loss_aver')
        for i in range(len_class):
            class_center_distance[i] = F.pairwise_distance(source_cluster_center[i].reshape(1, -1),
                                                           target_cluster_center[i].reshape(1, -1),
                                                           p=2).item()
    elif combine_pred.find('Cosine') != -1:
        print('using Cosine cal_cos_loss_aver')
        for i in range(len_class):
            class_center_distance[i] = 1 - torch.cosine_similarity(source_cluster_center[i], target_cluster_center[i],
                                                                   dim=0)
    else:
        print('ERROR.....cal_cos_loss_aver')
    shift_loss = class_center_distance.mean()
    return shift_loss, distance


def cal_cos_loss_source_cos(source_center, target_center, source_cluster_center, target_cluster_center,
                            combine_pred='Euclidean'):
    """
    计算源域和目标域上 每个类的中心到其他类的中心的cos距离 之差  class_num * (class_num)/2 个 的平均值
    :param source_center: 源域的全局中心
    :param target_center: 目标域的全局中心
    :param source_cluster_center: 源域的各类中心
    :param target_cluster_center: 目标域的各类中心
    :return: 每个类的中心到其他类的中心的cos距离 之差 的平均值， 每个类减去全局中心的对齐的loss值平均值
    """
    shift_source_cluster_center = source_cluster_center - source_center
    shift_target_cluster_center = target_cluster_center - source_center
    # print('shift_target_cluster_center.shape', shift_target_cluster_center.shape)
    len_class = shift_source_cluster_center.size(0)

    distance = 0.0
    for ii in range(len_class):
        for jj in range(ii, len_class):
            sij = torch.cosine_similarity(shift_source_cluster_center[ii], shift_source_cluster_center[jj], dim=0)
            tij = torch.cosine_similarity(shift_target_cluster_center[ii], shift_target_cluster_center[jj], dim=0)
            # print(type(sij))
            distance += torch.abs(sij - tij)
    distance /= (ii * (ii - 1) / 2)

    class_center_distance = torch.zeros(len_class).cuda()

    if combine_pred.find('Euclidean') != -1:
        # print('using Euclidean cal_cos_loss_aver')
        for i in range(len_class):
            class_center_distance[i] = F.pairwise_distance(source_cluster_center[i].reshape(1, -1),
                                                           target_cluster_center[i].reshape(1, -1),
                                                           p=2).item()
    elif combine_pred.find('Cosine') != -1:
        # print('using Cosine cal_cos_loss_aver')
        for i in range(len_class):
            class_center_distance[i] = 1 - torch.cosine_similarity(source_cluster_center[i], target_cluster_center[i],
                                                                   dim=0)
    else:
        print('ERROR.....cal_cos_loss_aver')
    shift_loss = class_center_distance.mean()
    return shift_loss, distance


def cal_cos_loss_aver_source(source_center, target_center, source_cluster_center, target_cluster_center):
    """
    计算源域和目标域上 每个类的中心到其他类的中心的cos距离 之差  class_num * (class_num)/2 个 的平均值
    都减去source的中心
    :param source_center: 源域的全局中心
    :param target_center: 目标域的全局中心
    :param source_cluster_center: 源域的各类中心
    :param target_cluster_center: 目标域的各类中心
    :return: 每个类的中心到其他类的中心的cos距离 之差 的平均值， 每个类减去全局中心的对齐的loss值平均值
    """
    shift_source_cluster_center = source_cluster_center - source_center
    shift_target_cluster_center = target_cluster_center - source_center
    # print('shift_target_cluster_center.shape', shift_target_cluster_center.shape)
    len_class = shift_source_cluster_center.size(0)

    distance = 0.0
    for ii in range(len_class):
        for jj in range(ii, len_class):
            sij = torch.cosine_similarity(shift_source_cluster_center[ii], shift_source_cluster_center[jj], dim=0)
            tij = torch.cosine_similarity(shift_target_cluster_center[ii], shift_target_cluster_center[jj], dim=0)
            # print(type(sij))
            distance += torch.abs(sij - tij)
    distance /= (ii * (ii - 1) / 2)

    class_center_distance = torch.zeros(len_class).cuda()
    for i in range(len_class):
        class_center_distance[i] = F.pairwise_distance(source_cluster_center[i].reshape(1, -1),
                                                       target_cluster_center[i].reshape(1, -1),
                                                       p=2).item()
    shift_loss = class_center_distance.mean()
    return shift_loss, distance


def cal_cos_loss_1(source_center, target_center, source_cluster_center, target_cluster_center):
    """
    计算源域和目标域上 每个类的中心到其他类的中心的cos距离 之差  class_num * (class_num)/2 个 的和
    :param source_center: 源域的全局中心
    :param target_center: 目标域的全局中心
    :param source_cluster_center: 源域的各类中心
    :param target_cluster_center: 目标域的各类中心
    :return: 每个类的中心到其他类的中心的cos距离 之差 的和， 每个类减去全局中心的对齐的loss值平均值
    """
    shift_source_cluster_center = source_cluster_center - source_center
    shift_target_cluster_center = target_cluster_center - target_center
    len_class = shift_source_cluster_center.size(0)

    distance = 0.0
    for ii in range(len_class):
        for jj in range(ii, len_class):
            sij = torch.cosine_similarity(shift_source_cluster_center[ii], shift_source_cluster_center[jj], dim=0)
            tij = torch.cosine_similarity(shift_target_cluster_center[ii], shift_target_cluster_center[jj], dim=0)
            # print(type(sij))
            distance += torch.abs(sij - tij)

    class_center_distance = torch.zeros(len_class).cuda()
    for i in range(len_class):
        class_center_distance[i] = 1 - torch.cosine_similarity(shift_source_cluster_center[i],
                                                               shift_target_cluster_center[i], dim=0)
    shift_loss = class_center_distance.mean()
    return shift_loss, distance


def compute_cluster_center(features, labels, class_num):
    """
    计算特征中心
    :param features: 特征 n*dim
    :param labels: 特征的label n
    :param class_num: 类别数目 class_num
    :return: 特征的每个类的中心 class_num*dim
    """
    feature_dim = features.size(1)
    feature_centers = torch.zeros((class_num, feature_dim)).cuda()
    for i in range(class_num):
        feature_centers[i] = features[labels == i, :].mean(0)
    return feature_centers


def cal_git_loss(source_learned_feature, source_label, l_target_learned_feature,
                 l_target_label, source_cluster_center, target_cluster_center):
    """
    计算 gitloss 每个样本到自己的类的中心的距离/到其他类的距离之和
    :param source_learned_feature: 源域
    :param source_label: 源域的label
    :param l_target_learned_feature: 目标域有label
    :param l_target_label: 目标域的label
    :param source_cluster_center: 源域中心
    :param target_cluster_center: 目标域中心
    :return:
    """
    git_loss = 0.0
    s_len = source_learned_feature.size(0)
    l_len = l_target_learned_feature.size(0)
    class_num = source_cluster_center.size(0)
    s_l_len = s_len + l_len
    s_dist = source_learned_feature.mm(source_cluster_center.t())
    l_dist = l_target_learned_feature.mm(target_cluster_center.t())
    s_dist_sum1 = torch.sum(s_dist, dim=1)
    l_dist_sum1 = torch.sum(l_dist, dim=1)
    for ss in range(s_len):
        max_index = int(source_label[ss])
        # print('s_inner_cos_sim', s_dist[ss, max_index])
        # print('s_intra_cos_sim', (s_dist_sum1[ss] - s_dist[ss, max_index]))
        up = 1 - s_dist[ss, max_index]
        down = (class_num - 1) - (s_dist_sum1[ss] - s_dist[ss, max_index])
        git_loss += up / down
    for ll in range(l_len):
        max_index = int(l_target_label[ll])
        # print('l_inner_cos_sim', l_dist[ll, max_index])
        # print('l_intra_cos_sim', (l_dist_sum1[ll] - l_dist[ll, max_index]))
        up = 1 - l_dist[ll, max_index]
        down = (class_num - 1) - (l_dist_sum1[ll] - l_dist[ll, max_index])
        git_loss += up / down
    git_loss = git_loss / s_l_len
    return git_loss


#  make dirs for model_path, result_path, log_path, diagram_path
def make_dirs(args):
    """
    建立文件夹 老项目带过来的 没看
    :param args:
    :return:
    """
    save_name = '_'.join([args.source.lower(), args.target.lower()])
    log_path = os.path.join(args.checkpoint_path, 'logs')
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(log_path)
        print('Makedir: ' + str(log_path))
    args.log_path = os.path.join(log_path, save_name + '.txt')
    args.avg_path = os.path.join(log_path, save_name + '_avg.txt')


def k_means(target_features, target, c):
    eps = 1e-6
    cluster_iter = 5
    class_num = c.size(0)
    c_tar = c.data.clone()
    for itr in range(cluster_iter):
        dist_xt_ct_temp = target_features.unsqueeze(1) - c_tar.unsqueeze(0)
        dist_xt_ct = dist_xt_ct_temp.pow(2).sum(2)
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        # prec1 = accuracy(-1 * dist_xt_ct.data, target, topk=(1, ))[0].item()
        # print('1 accuracy', prec1)
        # is_best = prec1 > best_prec
        c_tar_temp = torch.cuda.FloatTensor(class_num, c_tar.size(1)).fill_(0)
        count = torch.cuda.FloatTensor(class_num, 1).fill_(0)
        for k in range(class_num):
            c_tar_temp[k] += target_features[idx_sim.squeeze(1) == k].sum(0)
            count[k] += (idx_sim.squeeze(1) == k).float().sum()
        c_tar_temp /= (count + eps)
        c_tar = c_tar_temp.clone()
        del dist_xt_ct_temp
        torch.cuda.empty_cache()
    return c_tar


def kernel_kmeans(target_features, pseudo_labels, targets, configuration, cluster_iter, epoch):
    # define kernel k-means clustering
    class_num = configuration['class_number']
    eps = 1e-6
    cluster_kernel = 'rbf'
    gamma = None
    kkm = KernelKMeans(n_clusters=class_num, max_iter=cluster_iter, random_state=0, kernel=cluster_kernel, gamma=gamma,
                       verbose=1)
    kkm.fit(np.array(target_features.detach().cpu()), initial_label=np.array(pseudo_labels.long().cpu()),
            true_label=targets.numpy(), epoch=epoch)  # .max(1)[1].
    idx_sim = torch.from_numpy(kkm.labels_)
    c_tar = torch.cuda.FloatTensor(class_num, target_features.size(1)).fill_(0)
    count = torch.cuda.FloatTensor(class_num, 1).fill_(0)
    print('target_size', targets.size(0))
    for i in range(targets.size(0)):
        c_tar[idx_sim[i]] += target_features[i]
        count[idx_sim[i]] += 1
    c_tar /= (count + eps)
    prec1 = kkm.prec1_
    print('refine accuracy', prec1)
    # print('acc', idx_sim.eq(targets).sum()/ targets.size(0))
    torch.cuda.empty_cache()
    return c_tar, idx_sim


def spherical_k_means(target_features, targets, c):
    eps = 1e-6
    cluster_iter = 1
    class_num = c.size(0)
    c_tar = c.data.clone()
    for itr in range(cluster_iter):
        torch.cuda.empty_cache()
        dist_xt_ct_temp = target_features.unsqueeze(1) * c_tar.unsqueeze(0)
        dist_xt_ct = 0.5 * (1 - dist_xt_ct_temp.sum(2) / (
                target_features.norm(2, dim=1, keepdim=True) * c_tar.norm(2, dim=1, keepdim=True).t() + eps))
        _, idx_sim = (-1 * dist_xt_ct).data.topk(1, 1, True, True)
        # prec1 = accuracy( -1 * dist_xt_ct.data, targets, topk=(1, ))[0].item()
        # print('refine accuracy', prec1)
        c_tar_temp = torch.cuda.FloatTensor(class_num, c_tar.size(1)).fill_(0)
        for k in range(class_num):
            c_tar_temp[k] += (target_features[idx_sim.squeeze(1) == k] / (
                    target_features[idx_sim.squeeze(1) == k].norm(2, dim=1, keepdim=True) + eps)).sum(0)
        c_tar = c_tar_temp.clone()

        del dist_xt_ct_temp
        gc.collect()
        torch.cuda.empty_cache()
    del target_features
    del targets
    torch.cuda.empty_cache()
    return c_tar, idx_sim


def getNumber(total=800, total_step=10, step_index=0):
    """
    获取所在批次应该挑选的unlabeled target 的样本的个数
    :param total:
    :param total_step:
    :param step_index:
    :return:
    """
    one_step = total // total_step
    if step_index == (total_step - 1):
        print('get topk', total)
        return total
    print('get topk', (step_index + 1) * one_step)
    return (step_index + 1) * one_step
