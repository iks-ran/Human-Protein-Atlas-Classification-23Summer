import os
import json
import argparse
import torch

def get_hparams():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                        help='JSON file for configuration')
    parser.add_argument('-r', '--resume', type=str, default='',
                        help='Path of model to resume')
    args = parser.parse_args()

    if args.resume == '':
        config_path = os.path.abspath(args.config)
    else:
        config_path = torch.load(args.resume)['config_path']
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(config_path, resume=args.resume, **config)
    print(hparams)

    return hparams

class HyperParams(object):

    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HyperParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

class HParams(HyperParams):

    def __init__(self, config_path="./configs/config.json", resume='' , **kwargs):
        super(HParams, self).__init__(**kwargs)

        self.resume = resume
        self.config_path = config_path
        print(f'Successfully init configuration form \'{config_path}\'!')