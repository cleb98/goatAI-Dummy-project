import random
import socket
import os
import requests
import zipfile
import numpy as np
import torch
import yaml
from path import Path

from typing import Any
from typing import Dict
from typing import Optional



''' Summary
- **`random`, `numpy`, and `torch`**: These libraries are used to set random seeds across different modules, which is crucial for reproducibility.
- **`socket`**: Used to get the hostname of the machine, which is useful for logging, especially when running experiments on different systems.
- **`yaml`**: This helps in loading configuration settings, making the script more dynamic and easier to adapt to changes.
- **`path`**: Handles file paths elegantly, making it easy to switch between directories.

#### Use Case
The script is particularly useful for managing experiments in machine learning. It provides a robust way to:
1. Set up an experiment environment by configuring paths, devices, and hyperparameters.
2. Log important metadata like the machine's hostname.
3. Ensure reproducibility by controlling random seed initialization.

This kind of utility is commonly used in research settings and startups to streamline the process of setting up and running different configurations, ensuring experiments are reproducible and traceable across different systems.
'''


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)




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

        #to fix the download feature
        # self.url = y.get('DS_URL', None)  # type: str
        # if not self.ds_root.exists():
        #     print( f'directory `DS_ROOT={self.ds_root}` does not exist')
        #     Path(self.ds_root).makedirs()
        #     self.dataset_download()
        #     print(f'dataset downloaded to {self.ds_root}')

    # def dataset_download(self):
    #     # Assicurati che la cartella di destinazione esista
    #     dest_folder = self.ds_root
    #     os.makedirs(dest_folder, exist_ok=True)
    #
    #     # Estrai l'ID del file dall'URL di Google Drive
    #     file_id = self.url.split('/d/')[1].split('/')[0]
    #     session = requests.Session()
    #     download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    #
    #     # Invia una richiesta iniziale per ottenere il token di conferma
    #     response = session.get(download_url, params={'id': file_id}, stream=True)
    #     token = get_confirm_token(response)
    #
    #     if token:
    #         params = {'id': file_id, 'confirm': token}
    #         response = session.get(download_url, params=params, stream=True)
    #
    #     # Definisci il percorso per il file ZIP scaricato
    #     zip_path = os.path.join(dest_folder, 'downloaded.zip')
    #
    #     # Scarica il file ZIP
    #     save_response_content(response, zip_path)
    #
    #     # Estrai il contenuto del file ZIP
    #     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    #         zip_ref.extractall(dest_folder)
    #
    #     # Elimina il file ZIP
    #     os.remove(zip_path)

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
