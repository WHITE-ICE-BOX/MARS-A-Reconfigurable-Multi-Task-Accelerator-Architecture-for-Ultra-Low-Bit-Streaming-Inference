import os
import sys
from configparser import ConfigParser

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from .CNV_param import cnv_param

CFG_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'cfg_bitwidth'))


def get_model_cfg_bitwidth(name):
    cfg = ConfigParser()
    config_path = os.path.join(CFG_DIR, name.lower() + '.ini')
    assert os.path.exists(config_path), f"{config_path} not found."
    cfg.read(config_path)
    return cfg


def model_with_cfg_bitwidth(name):
    cfg = get_model_cfg_bitwidth(name)
    model = cnv_param(cfg)
    return model, cfg
