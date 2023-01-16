from asyncio import base_events
import json
import yaml
import os
import glob
import sys
from pathlib import Path
from dotmap import DotMap
from shutil import copyfile
from natsort import os_sorted

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from definitions import ROOT_DIR


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    """
    for dir_ in dirs:
        Path(dir_).mkdir(parents=True, exist_ok=True)

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def get_config_from_yaml(yaml_file):
    """
    Get the config from a yaml file
    :param yaml_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(yaml_file, 'r') as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.Loader)

    # convert the dictionary to a namespace using bunch lib
    config = DotMap(config_dict)

    return config, config_dict


def get_config(file_path):
    if file_path.endswith('.yaml') or file_path.endswith('.yml'):
        return get_config_from_yaml(file_path)
    elif file_path.endswith('.json'):
        return get_config_from_json(file_path)


def process_config(file, name, output_dir=ROOT_DIR, dirs=True, config_copy=True):
    print('Processing config..')
    config, _ = get_config(file)
    
    config.exp.name = name
    
    if(config.exp.name):
        base_dir = os.path.join(output_dir, config.exp.name)
        config.experiment_dir = base_dir
        config.callbacks.tensorboard_log_dir = os.path.join(base_dir, "logs/")
        
        new_version = 0
        if os.path.exists(config.callbacks.tensorboard_log_dir):
            # Get experiment version
            version = os_sorted(glob.glob(os.path.join(config.callbacks.tensorboard_log_dir, 'version_*')))
            if len(version) > 0:
                new_version = int(os.path.basename(version[-1]).split('_')[-1]) + 1
        config.callbacks.checkpoint_dir = os.path.join(base_dir, "checkpoints", f'version_{new_version}')

        config.callbacks.backup_dir = os.path.join(base_dir, "backup/")
        if(config.mode == 'train'):
            config.results.performance_dir = os.path.join(base_dir, "results/online")
            config_dir = config.callbacks.checkpoint_dir.replace('/checkpoints/', '/configs/')
            config.callbacks.config_dir = config_dir
            if dirs:
                create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir, config.results.performance_dir, config_dir])
            if config_copy:
                copyfile(file, os.path.join(config_dir, Path(file).name))
                
        elif(config.mode == 'eval'):
            dirname = os.path.dirname(os.path.dirname(os.path.dirname(config.tester.checkpoint_path)))
            checkpoint = os.path.basename(config.tester.checkpoint_path)
            config.results.performance_dir = os.path.join(dirname,'results','offline',checkpoint)
            if dirs:
                create_dirs([config.results.performance_dir])
            if config_copy:
                copyfile(file, os.path.join(config.results.performance_dir,'config.json'))
                
    return config

