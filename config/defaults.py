from yacs.config import CfgNode as CN

_C = CN()

_C.EXP_ROOT = "exp"
_C.TB_ROOT = "tb_runs"
_C.RUNS_NAME = "debug"
_C.SEED = 777

# model realted
_C.MODEL = CN()
_C.MODEL.NAME = "LDF"
_C.MODEL.BAKCBONE_PATH = "checkpoints/resnet50-19c8e357.pth"
_C.MODEL.WARMUP_PATH = ""
_C.MODEL.DETECTOR_PATH = ""


# dataset related 
_C.DATA = CN()
_C.DATA.TRAIN_BATCH_SIZE = 32
_C.DATA.TEST_BATCH_SIZE = 32
# dataet loader pseudo label generation dataset
_C.DATA.PSE_BATCH_SIZE = 32

_C.DATA.NUM_WORKER = 4
_C.DATA.SRC_DATAPATH = 'data/SYNSOD'
_C.DATA.TGT_DATAPATH = 'data/DUTS'
_C.DATA.TGT_DATASET = 'duts_tr'
_C.DATA.SRC_DATASET = 'synsod'

_C.DATA.DATAROOT = 'data'

## augmentation
_C.DATA.STRONG_AUG = False # Trainig with strong augmentation

### elastic transform
_C.DATA.ET_ALPHA = 120 
_C.DATA.ET_SIGMA = 120 * 0.06
_C.DATA.ET_AFFINE = 120 * 0.03
_C.DATA.ET_P = 0.5 


### random size crop 
_C.DATA.RC_SCALE = [0.875, 1] # Trainig strong augmentation

### fda aug
_C.DATA.FDA_BETA = 0.01 # Trainig strong augmentation
_C.DATA.FDA_P = 0.5 # Trainig strong augmentation


## color jitter
_C.DATA.CJ_P = 0.5
_C.DATA.CJ_B = 0.4
_C.DATA.CJ_C = 0.4
_C.DATA.CJ_S = 0.4
_C.DATA.CJ_H = 0


# preprocessing
_C.DATA.TRAIN_MUTI_SCALE = [224, 256, 288, 320, 352]
_C.DATA.TRAIN_SHAPE = [352, 352]
_C.DATA.TEST_SHAPE = [352, 352]
_C.DATA.PIXEL_MEAN = [124.55, 118.90, 102.94] # rgb channel
_C.DATA.PIXEL_STD = [ 56.77,  55.97,  57.50]


# model training
_C.SOLVER = CN()
_C.SOLVER.LR = 0.05
_C.SOLVER.BASE_LR = 0.05 * 0.1

_C.SOLVER.DECAY = 5e-4
_C.SOLVER.MOMEN = 0.9
_C.SOLVER.AMP = True
_C.SOLVER.BETAS = [0.9, 0.99]


## iterative training related
_C.SOLVER.EPOCH_PER_ROUND = 36 # epoch per round 
_C.SOLVER.ITER_PER_EPOCH = 256
_C.SOLVER.ROUND_NUM = 7 # warmup round is included
_C.SOLVER.WARMUP_EPOCH = 48
_C.SOLVER.LABEL_SELECT_STRATEGY = 'consis'
_C.SOLVER.REINIT_HEAD = False


## pseudo label updating
_C.SOLVER.PSEUDO_UPDATE_POLICY = 'src_half'
_C.SOLVER.PSE_POLICY = 'portion'
_C.SOLVER.SRC_INIT_PORTION = 0.5
_C.SOLVER.TGT_INIT_PORTION = 0.1
_C.SOLVER.TGT_MAX_PORTION = 0.6

_C.SOLVER.VAR_THRESHOLD = 0.002

### filter out top portion of data
_C.SOLVER.PSE_FILTER_PORTION = 0.05
_C.SOLVER.ONLY_MASK = True
### pixel weight
_C.SOLVER.PIXEL_WEIGHT = 20
_C.SOLVER.ONE_HOT = True # binarize the mask

## learning rate scheduler
#['lin_epoch', 'lin_step', 'cyc', 'cos', 'multi_cos', 'multi_step']
# misc.py:get_scheduler for more detail
_C.SOLVER.SCHEDULER = 'cyc' # scheduler type
_C.SOLVER.LR_DECAY_RATE = 0.75 # learning rate decay for each round


## consistency 
_C.SOLVER.AUG_TYPE = 'fuse'
_C.SOLVER.FDA_NUM = 1
_C.SOLVER.RAND_SCALE_NUM = 1
_C.SOLVER.FLIP_NUM = 1

_C.SOLVER.REINIT_HEAD = False

## heuristic filtering 
# filter out pseudo label with saliency reigion smaller than 0.01 or bigger than 0.99
_C.SOLVER.B_THRES = 0.01 # black threshold
_C.SOLVER.W_THRES = 0.99 # white threshold


## log
_C.SOLVER.PRINT_FREQ = 32
# _C.SOLVER.img_record_interval = 5000
_C.SOLVER.IMG_RECORD_INTERVAL = 5000



_C.TEST = CN()
_C.TEST.EVAL_INTERVAL = 2
# _C.TEST.SAVE_MASK_PATH = ""
# _C.TEST.SAVE_BODY_PATH = ""
# _C.TEST.SAVE_DETAIL_PATH = ""

_C.TEST.ALL_DS_NAME = [
        'duts_te', 
        'duts_val',
        'duts_om', 
        'hkuis',
        'eccsd',
        'thur',
        'sod',
        'pascal_s',
        'msra_b',
    ]

_C.TEST.DS_EVAL_TRAIN = [
    'duts_te', 
    'thur'
] # datasets eval during training

_C.TEST.METRICS_EVAL_TRAIN = [
    'mae', 
    'sm'
] 

# _C.TEST.DS_EVAL_FINAL = [ # evaluation at the end of training
#         'duts_te',
#         'eccsd',
#         'duts_om',
#         'hkuis',
#         # 'thur',
#         'sod',
#         # 'pascal_s',
#         'msra_b',
#     ]

_C.TEST.EVAL_METRICS = [
    'fm', 
    'wfm', 
    'sm', 
    'em', 
    'mae'
]

_C.TEST.EVAL_REPORT_METRICS = [
    'sm', 
    'wfm', 
    'adp_fm', 'mean_fm', 'max_fm', 
    'adp_em', 'mean_em', 'max_em', 
    'mae'
]

_C.TEST.EVAL_DATASET = [ # evaluation at the end of run
        'duts_te',
        'eccsd',
        'duts_om',
        'hkuis',
        # 'thur',
        'sod',
        'pascal_s',
        # 'msra_b',
    ]

_C.TEST.MODEL_ROOTS = ""
_C.TEST.CHECKPOINT = ""
_C.TEST.SNAPSHOT = ""



# _C.TEST.TEST_CONFIG = {
#     'duts_te': {'gt_path':'data/DUTS'},
#     'duts_om': {'gt_path':'data/DUT-OMRON'},
#     'thur': {'gt_path':'data/THUR15K'},
#     'eccsd': {'gt_path':'data/ECSSD'},
#     'hkuis': {'gt_path':'data/HKU-IS'},
#     'sod': {'gt_path':'data/SOD'},
#     'pascal_s': {'gt_path':'data/PASCAL-S'},
#     'msra_b': {'gt_path':'data/MSRA-B'},
# }



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()