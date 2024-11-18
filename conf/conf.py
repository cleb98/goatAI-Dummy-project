import random
import socket
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
import yaml
from path import Path


def set_seed(seed=None):
    # type: (Optional[int]) -> int
    """
    Set the random seed using the required value (`seed`)
    or a random value if `seed` is `None`.
    :return: the newly set seed
    """
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


class Conf(object):
    HOSTNAME = socket.gethostname()


    @property
    def dict(self):
        # type: () -> Dict[str, Any]
        """
        :return: dictionary version of the configuration file
        """
        x = self.__dict__
        y = {}
        for key in x:
            if key not in self.keys_to_hide:
                y[key] = x[key]
        return y


    def __init__(self, cnf_path=None, seed=None, exp_name=None, log=True):
        # type: (str, int, str, bool) -> None
        """
        :param cnf_path: optional path of the configuration file
        :param seed: desired seed for the RNG;
            >> if `None`, it will be chosen randomly
        :param exp_name: name of the experiment
        :param log: `True` if you want to log each step; `False` otherwise
        """
        self.exp_name = exp_name
        self.log_each_step = log

        # print project name and host name
        project_name = Path(__file__).parent.parent.basename()
        m_str = f'┃ {project_name}@{Conf.HOSTNAME} ┃'
        u_str = '┏' + '━' * (len(m_str) - 2) + '┓'
        b_str = '┗' + '━' * (len(m_str) - 2) + '┛'
        print(u_str + '\n' + m_str + '\n' + b_str)

        # define output paths
        self.project_root = Path(__file__).parent.parent
        self.project_log_path = self.project_root.absolute() / 'log'
        self.exp_log_path = self.project_log_path / exp_name

        # set random seed
        self.seed = set_seed(seed)  # type: int

        self.keys_to_hide = list(self.__dict__.keys()) + ['keys_to_hide']

        # if the configuration file is not specified
        # try to load a configuration file based on the experiment name
        tmp = Path(__file__).parent / (self.exp_name + '.yaml')
        if cnf_path is None and tmp.exists():
            cnf_path = tmp

        # read the YAML configuration file
        if cnf_path is None:
            y = {}
        else:
            conf_file = open(cnf_path, 'r')
            y = yaml.load(conf_file, Loader=yaml.Loader)


        # read configuration parameters from YAML file
        # or set their default value
        self.lr = y.get('LR', 0.0001)  # type: float
        self.epochs = y.get('EPOCHS', 32)  # type: int
        self.n_workers = y.get('N_WORKERS', 4)  # type: int
        self.batch_size = y.get('BATCH_SIZE', 8)  # type: int
        self.max_patience = y.get('MAX_PATIENCE', 8)  # type: int

        default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = y.get('DEVICE', default_device)  # type: str
        print(f'▶ using device: {self.device}')

        self.ds_root = y.get('DS_ROOT', None)  # type: str

        assert self.ds_root, \
            f'you must specify the `DS_ROOT` parameter ' \
            f'in the configuration file'

        self.ds_root = self.ds_root.replace(
            '$PROJECT_DIR', self.project_root
        )
        self.ds_root = Path(self.ds_root)
        assert self.ds_root.exists(), \
            f'directory `DS_ROOT={self.ds_root}` does not exist'


    def __str__(self):
        # type: () -> str
        """
        :return: string representation of the configuration object;
            ->> NOTE: this is a color-coded string
        """
        out_str = ''
        __d = self.dict
        for key in __d:
            value = __d[key]
            if isinstance(value, str):
                value = f'\'{value}\''
            out_str += f'{key.upper()}: {value}\n'
        return out_str[:-1]


def show_default_params():
    """
    Print default configuration parameters
    """
    cnf = Conf(exp_name='default')
    print(f'\nDefault configuration parameters: \n{cnf}')


if __name__ == '__main__':
    show_default_params()
