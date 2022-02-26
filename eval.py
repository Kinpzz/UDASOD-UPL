import logging
import os
import argparse

from lib.evalualtion import eval
from config.defaults import get_cfg_defaults
from lib.misc import get_time_str, setup_logger
from lib.pipeline_ops import setup_test_dataset_config
import copy


def parse_aug():
    parser = argparse.ArgumentParser(description='DA_USOD evaluation')
    parser.add_argument('--exp_config', type=str, default='', help='configuration dir')
    parser.add_argument('--pred_dir', type=str, default='', help='prediction dir')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    args = parser.parse_args()
    return args

def do_eval(pred_dir, cfg):
    # test_loaders = {ds: test_config[ds]['loader'] for ds in ds_names}
    ds_names = cfg.TEST.EVAL_DATASET
    eval_config = setup_test_dataset_config(cfg)
    fin_res = eval(
        pred_dir,
        eval_config,
        ds_names,
        cfg.TEST.EVAL_METRICS,
    )

    # header line
    def get_csv_header_line(datasets, metrics):
        """
        args:
        - 
        return:
        - head_lines: two header line
        """
        headerline1 = [''] + [ds for ds in datasets for _ in metrics]
        headerline2 = [''] + [met for _ in datasets for met in metrics]
        return headerline1, headerline2

    headerlines = get_csv_header_line(cfg.TEST.EVAL_DATASET,cfg.TEST.EVAL_REPORT_METRICS)
    with open(cfg.out_csv_res, 'a') as f:
        f.write('\n'.join([','.join(hl) for hl in headerlines]))
        f.write('\n')
        f.write(f',')
        for ds in ds_names:
            f.write(','.join([str(fin_res[ds][m]) for m in cfg.TEST.EVAL_REPORT_METRICS]))
            f.write(',')
        f.write('\n')


if __name__ == "__main__":

    args = parse_aug()

    cfg = get_cfg_defaults()
    
    if args.exp_config != "":
        cfg.merge_from_file(args.exp_config)
    
    # merge from command line
    cfg.merge_from_list(args.opts)

    cfg_for_dump = copy.deepcopy(cfg)
    cfg = args
    for attr_str in cfg_for_dump:
        cfg.__setattr__(attr_str, cfg_for_dump.__getattr__(attr_str))
    
    cfg.savepath = f'eval_res_{get_time_str()}'

    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    
    ## config file
    with open(os.path.join(cfg.savepath, 'eval_config.yaml'), 'w') as f:
        f.write(cfg_for_dump.dump()) # use cfg for dump we can resotre the env 
    
    cfg.out_csv_res = os.path.join(cfg.savepath, 'eval.csv')

    setup_logger(cfg.savepath, 'eval')
    logging.info(f"eval prediction dataset: {cfg.pred_dir}")

    do_eval(cfg.pred_dir, cfg)

    