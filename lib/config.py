import os
import sys
from easydict import EasyDict
from ast import literal_eval
import copy
import yaml


CONF = EasyDict()

# ============================ PATHS ============================ #
# Model Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "./" # TODO: Remember to update this to the correct path
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")
CONF.PATH.WEIGHTS = os.path.join(CONF.PATH.BASE, "weights")
CONF.PATH.SQATASK = "./dataset/sqa3d/SQA3D/assets/data/sqa_task/" # TODO: Remember to update this to the correct path
CONF.PATH.DATA = "./dataset/sqa3d/SQA3D/ScanQA/data/"  # TODO: Remember to update this to the correct path
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "scannet")
# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")
CONF.PATH.SQA = os.path.join(CONF.PATH.DATA, "qa")
CONF.SCANNET_FRAMES_ROOT = os.path.join(CONF.PATH.DATA, "frames_square")
CONF.ENET_FEATURES_ROOT = os.path.join(CONF.PATH.DATA, "enet_features")
CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
CONF.SCENE_NAMES = sorted(os.listdir(CONF.PATH.SCANNET_SCANS))
CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.DATA, "scannetv2_enet.pth")
CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool")
CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")
# scannet meta
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "scannetv2.txt")
CONF.SCANNETV2_LIST_DEV_1 = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_dev_1.txt")
CONF.SCANNETV2_LIST_DEV_2 = os.path.join(CONF.PATH.SCANNET_META, "scannetv2_dev_2.txt")
# output
CONF.PATH.OUTPUT = os.path.join(CONF.PATH.DATA, "outputs")  # TODO: Remember to update this to the correct path

# ============================ TRAIN ============================ #
# train and val
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_TEXT_LEN = 100 #
CONF.TRAIN.SEED = 0

CONF.TRAIN.NO3D = False
CONF.TRAIN.USE_SITUATION = True
CONF.TRAIN.PREDICT_SITUATION = True
CONF.TRAIN.SITUATION_CLASS = True
CONF.TRAIN.USE_AUGMENTATION = True

CONF.TRAIN.STA_VEC_VISION_ADD = True # not used anymore
CONF.TRAIN.no_update_filter = ["openscene_net"] # ["openscene_net"]


# visualization
CONF.VISUALIZATION = EasyDict()
CONF.VISUALIZATION.SAVEOBJ = False
CONF.VISUALIZATION.TRAIN = False
CONF.VISUALIZATION.VAL = False

# loss function
CONF.LOSS = EasyDict()
CONF.LOSS.VOTE_W = 1.0
CONF.LOSS.OBJECTNESS_W = 0.5
CONF.LOSS.BOX_W = 1.0
CONF.LOSS.SEM_CLS_W = 0.1
CONF.LOSS.SITUATION_W = 0.1
CONF.LOSS.QA_W = 0.1
CONF.LOSS.SITUATION_POS_W = 1.0
CONF.LOSS.SITUATION_ROT_W = 1.0

# augmentation
CONF.AUGMENTATION = EasyDict()
CONF.AUGMENTATION.NO_MIRROR = True
CONF.AUGMENTATION.NO_ROTX = True
CONF.AUGMENTATION.NO_ROTY = True
CONF.AUGMENTATION.NO_ROTZ = False 
CONF.AUGMENTATION.NO_TRANS = True


# ============================ OpenScene ============================ #
CONF.OPENSCENE = EasyDict()
CONF.OPENSCENE.voxel_size = 0.02
CONF.OPENSCENE.data_root = "./dataset/ScanNet/openscene/scannet_3d" # TODO: Remember to update this to the correct path
CONF.OPENSCENE.feature_2d_extractor = "openseg"

if CONF.OPENSCENE.feature_2d_extractor == 'openseg':
    CONF.OPENSCENE.model_url = 'https://cvg-data.inf.ethz.ch/openscene/models/scannet_openseg.pth.tar'
elif CONF.OPENSCENE.feature_2d_extractor == 'lseg':
    CONF.OPENSCENE.model_url = 'https://cvg-data.inf.ethz.ch/openscene/models/scannet_lseg.pth.tar'
else:
    raise NotImplementedError

CONF.OPENSCENE.classes = 20
CONF.OPENSCENE.num_points = 256 # 256 normally, but 512 for visualization
CONF.OPENSCENE.feat_dim = 256
CONF.OPENSCENE.arch_3d = 'MinkUNet18A'
CONF.OPENSCENE.ignore_label = 255
CONF.OPENSCENE.workers = 8
CONF.OPENSCENE.power = 0.9
CONF.OPENSCENE.momentum = 0.9
CONF.OPENSCENE.print_freq = 10
CONF.OPENSCENE.mark_no_feature_to_unknown = True
CONF.OPENSCENE.prompt_eng = True
CONF.OPENSCENE.final_result = False # True when testing, False when training
CONF.OPENSCENE.model_path = '/root/.cache/torch/hub/checkpoints/scannet_openseg.pth.tar'  # TODO: Remember to update this to the correct path
CONF.OPENSCENE.scale_augmentation_bound = (0.9, 1.1)
CONF.OPENSCENE.rotation_augmentation_bound = 18 # rand(0, 1)*pi/18 - pi/36
CONF.OPENSCENE.translation_augmentation_ratio_bound = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
CONF.OPENSCENE.elastic_distortion_params = ((0.2, 0.4), (0.8, 1.6))


class CfgNode(dict):
    '''
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    '''

    def __init__(self, init_dict=None, key_list=None):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, value in init_dict.items():
            if isinstance(value, dict):
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(value, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(seq_, num_spaces):
            seq = seq_.split("\n")
            if len(seq) == 1:
                return seq_
            first = seq.pop(0)
            seq = [(num_spaces * " ") + line for line in seq]
            seq = "\n".join(seq)
            seq = first + "\n" + seq
            return seq

        r = ""
        seq = []
        for k, value in sorted(self.items()):
            seperator = "\n" if isinstance(value, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(value))
            attr_str = _indent(attr_str, 2)
            seq.append(attr_str)
        r += "\n".join(seq)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file):
    '''Load from config files.'''

    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, value in cfg_from_file[key].items():
            cfg[k] = value

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg, cfg_list):
    '''Merge configs from a list.'''

    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg


def _decode_cfg_value(v):
    '''Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    '''
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, full_key):
    '''Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    '''

    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type or original is None:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )
