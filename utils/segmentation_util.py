import os
import shutil
from os.path import join
import glob
import numpy as np

import torch
from torch import nn
from PIL import Image
import open3d as o3d
import clip

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


UNKNOWN_ID = 255
NO_FEATURE_ID = 256


SCANNET_LABELS_20 = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                     'toilet', 'sink', 'bathtub', 'otherfurniture')

SCANNET_COLOR_MAP_20 = {
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    16: (219., 219., 141.),
    24: (255., 127., 14.),
    28: (158., 218., 229.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    36: (227., 119., 194.),
    39: (82., 84., 163.),
    0: (0., 0., 0.), # unlabel/unknown
}


def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        return confusion[:num_classes, :num_classes]

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)


def get_iou(label_id, confusion):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom


def evaluate(pred_ids, gt_ids, stdout=False, dataset='scannet_3d'):
    if stdout:
        print('evaluating', gt_ids.size, 'points...')
    if 'scannet_3d' in dataset:
        CLASS_LABELS = SCANNET_LABELS_20
    elif 'matterport_3d_40' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_40
    elif 'matterport_3d_80' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_80
    elif 'matterport_3d_160' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_160
    elif 'matterport_3d' in dataset:
        CLASS_LABELS = MATTERPORT_LABELS_21
    elif 'nuscenes_3d' in dataset:
        CLASS_LABELS = NUSCENES_LABELS_16
    else:
        raise NotImplementedError

    N_CLASSES = len(CLASS_LABELS)
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0

    count = 0
    for i in range(N_CLASSES):
        label_name = CLASS_LABELS[i]
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue

        class_ious[label_name] = get_iou(i, confusion)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    if stdout:
        print('classes          IoU')
        print('----------------------------')
        for i in range(N_CLASSES):
            label_name = CLASS_LABELS[i]
            try:
                if 'matterport' in dataset:
                    print('{0:<14s}: {1:>5.3f}'.format(label_name, class_accs[label_name]))

                else:
                    print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(
                        label_name,
                        class_ious[label_name][0],
                        class_ious[label_name][1],
                        class_ious[label_name][2]))
            except:
                print(label_name + ' error!')
                continue
        print('Mean IoU', mean_iou)
        print('Mean Acc', mean_acc)
    return mean_iou


def precompute_text_related_properties(labelset_name, feature_2d_extractor):
    '''pre-compute text features, labelset, palette, and mapper.'''

    if 'scannet' in labelset_name:
        labelset = list(SCANNET_LABELS_20)
        labelset[-1] = 'other' # change 'other furniture' to 'other'
        palette = get_palette(colormap='scannet')
    elif labelset_name == 'matterport_3d' or labelset_name == 'matterport':
        labelset = list(MATTERPORT_LABELS_21)
        palette = get_palette(colormap='matterport')
    elif 'matterport_3d_40' in labelset_name or labelset_name == 'matterport40':
        labelset = list(MATTERPORT_LABELS_40)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_80' in labelset_name or labelset_name == 'matterport80':
        labelset = list(MATTERPORT_LABELS_80)
        palette = get_palette(colormap='matterport_160')
    elif 'matterport_3d_160' in labelset_name or labelset_name == 'matterport160':
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')
    elif 'nuscenes' in labelset_name:
        labelset = list(NUSCENES_LABELS_16)
        palette = get_palette(colormap='nuscenes16')
    else: # an arbitrary dataset, just use a large labelset
        labelset = list(MATTERPORT_LABELS_160)
        palette = get_palette(colormap='matterport_160')

    mapper = None
    text_features = extract_text_feature(labelset, prompt_eng=True, labelset_name=labelset_name, feature_2d_extractor=feature_2d_extractor)
    labelset.append('unlabeled')
    return text_features, labelset, mapper, palette


def save_checkpoint(state, is_best, sav_path, filename='model_last.pth.tar'):
    filename = join(sav_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'model_best.pth.tar'))

def extract_clip_feature(labelset, model_name="ViT-B/32"):
    # "ViT-L/14@336px" # the big model that OpenSeg uses
    print("Loading CLIP {} model...".format(model_name))
    clip_pretrained, _ = clip.load(model_name, device='cuda', jit=False)
    print("Finish loading")

    if isinstance(labelset, str):
        lines = labelset.split(',')
    elif isinstance(labelset, list):
        lines = labelset
    else:
        raise NotImplementedError

    labels = []
    for line in lines:
        label = line
        labels.append(label)
    text = clip.tokenize(labels)
    text = text.cuda()
    text_features = clip_pretrained.encode_text(text)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features

def extract_text_feature(labelset, prompt_eng, labelset_name, feature_2d_extractor):
    '''extract CLIP text features.'''

    # a bit of prompt engineering
    if prompt_eng:
        print('Use prompt engineering: a XX in a scene')
        labelset = [ "a " + label + " in a scene" for label in labelset]
        if 'scannet_3d' in labelset_name:
            labelset[-1] = 'other'
        if 'matterport_3d' in labelset_name:
            labelset[-2] = 'other'
    if 'lseg' in feature_2d_extractor:
        text_features = extract_clip_feature(labelset)
    elif 'openseg' in feature_2d_extractor:
        text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px")
    else:
        raise NotImplementedError

    return text_features

def extract_clip_img_feature_from_folder(folder, model_name='ViT-L/14@336px'):
    '''extract CLIP image features from a folder of images.'''

    # "ViT-L/14@336px" # the big model that OpenSeg uses
    clip_pretrained, preprocess = clip.load(model_name, device='cuda', jit=False)

    img_paths = sorted(glob.glob(os.path.join(folder, "*")))
    img_feat = []
    for img_path in img_paths:
        image = Image.open(img_path)
        image_input = preprocess(image).unsqueeze(0).cuda()
        feat = clip_pretrained.encode_image(image_input).detach().cpu()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        img_feat.append(feat)

    img_feat = torch.cat(img_feat, dim=0)    
    return img_feat

class AverageMeter():
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    '''Sets the learning rate to the base LR decayed by 10 every step epochs'''
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    '''poly learning rate policy'''
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3, 4])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def export_pointcloud(name, points, colors=None, normals=None):
    if len(points.shape) > 2:
        points = points[0]
        if normals is not None:
            normals = normals[0]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        if normals is not None:
            normals = normals.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(name, pcd)

def export_mesh(name, v, f, c=None):
    if len(v.shape) > 2:
        v, f = v[0], f[0]
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    if c is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_triangle_mesh(name, mesh)

def visualize_labels(u_index, labels, palette, out_name, loc='lower left', ncol=7):
    patches = []
    for i, index in enumerate(u_index):
        label = labels[index]
        cur_color = [palette[index * 3] / 255.0, palette[index * 3 + 1] / 255.0, palette[index * 3 + 2] / 255.0]
        red_patch = mpatches.Patch(color=cur_color, label=label)
        patches.append(red_patch)
    plt.figure()
    plt.axis('off')
    legend = plt.legend(frameon=False, handles=patches, loc=loc, ncol=ncol, bbox_to_anchor=(0, -0.3), prop={'size': 5}, handlelength=0.7)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5,-5,5,5])))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(out_name, bbox_inches=bbox, dpi=300)
    plt.close()

def get_palette(num_cls=21, colormap='scannet'):
    if colormap == 'scannet':
        scannet_palette = []
        for _, value in SCANNET_COLOR_MAP_20.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    elif colormap == 'matterport':
        scannet_palette = []
        for _, value in MATTERPORT_COLOR_MAP_21.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    elif colormap == 'matterport_160':
        scannet_palette = []
        for _, value in MATTERPORT_COLOR_MAP_160.items():
            scannet_palette.append(np.array(value))
        palette = np.concatenate(scannet_palette)
    elif colormap == 'nuscenes16':
        nuscenes16_palette = []
        for _, value in NUSCENES16_COLORMAP.items():
            nuscenes16_palette.append(np.array(value))
        palette = np.concatenate(nuscenes16_palette)
    else:
        n = num_cls
        palette = [0]*(n*3)
        for j in range(0,n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while lab > 0:
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
    return palette

def convert_labels_with_palette(input, palette):
    '''Get image color palette for visualizing masks'''

    new_3d = np.zeros((input.shape[0], 3))
    u_index = np.unique(input)
    for index in u_index:
        if index == 255:
            index_ = 20
        else:
            index_ = index

        new_3d[input==index] = np.array(
            [palette[index_ * 3] / 255.0,
             palette[index_ * 3 + 1] / 255.0,
             palette[index_ * 3 + 2] / 255.0])

    return new_3d

class FocalLoss(nn.Module):

    def __init__(self, device, gamma=2, eps=1e-7, num_classes=20, reduce='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.num_classes = num_classes
        self.y = torch.eye(self.num_classes+1).to(device)
        self.reduce=reduce

    def forward(self, input, target):
        # y = one_hot(target, input.size(-1))
        target[target==255] = self.num_classes
        y = self.y[target]
        y = y[:, :self.num_classes]
        logit = input
        logit = logit.clamp(self.eps, 1. - self.eps)
        # logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        if self.reduce == 'mean':
            return loss.mean()
        else:
            return loss.sum()