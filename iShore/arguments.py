import argparse
import shutil
import os
import yaml

def get_parser():
    def process(**params):    # pass in variable numbers of args
        for key, value in params.items():
            parser.add_argument('--'+key, default=value, type=type(value))

    parser = argparse.ArgumentParser()
    
    with open('E:/point/Topo-boundary-master/coastline/config_dir.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf)
    
    with open('./config.yml', 'r') as f:
        conf = yaml.safe_load(f.read())    # load the config file
    process(**conf) 

    return parser

def check_and_add_dir(dir_path,clear=True):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            if clear:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)

def update_dir(args):
    check_and_add_dir('./checkpoints',clear=False)
    check_and_add_dir('./records/checkpoints',clear=False)
    check_and_add_dir('./records/tensorboard')
    check_and_add_dir('./records/train/vis/free_exploration')
    check_and_add_dir('./records/train/vis/restricted_exploration')
    check_and_add_dir('./records/valid/vis')
    check_and_add_dir('./records/valid/vertices_record')
    
def update_test_dir(args):
    check_and_add_dir('./records/test/skeleton')
    check_and_add_dir('./records/test/graph')
    check_and_add_dir('./records/test/final_vis')
    check_and_add_dir('./records/test/vertices_record')


def update_dir_seg(args):
    check_and_add_dir('./checkpoints',clear=False)
    check_and_add_dir('./records/seg/tensorboard')
    check_and_add_dir('./records/seg/valid')