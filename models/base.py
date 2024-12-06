from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import torch
from path import Path
from torch import nn


ConfDict = Dict[str, Any]


class BaseModel(nn.Module, metaclass=ABCMeta):

    def __init__(self):
        # type: () -> None
        super().__init__()
        self.cnf_dict = None  # type: Optional[ConfDict]


    @abstractmethod
    def forward(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.

        :param x: input tensor
        """
        ...


    @property
    def n_param(self):
        # type: () -> int
        """
        :return: number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    @property
    def device(self):
        # type: () -> str
        """
        Check the device on which the model is currently located.

        :return: string that represents the device on which the model
            is currently located
            ->> e.g.: 'cpu', 'cuda', 'cuda:0', 'cuda:1', ...
        """
        return str(next(self.parameters()).device)


    @property
    def is_cuda(self):
        # type: () -> bool
        """
        Check if the model is on a CUDA device.

        :return: `True` if the model is on CUDA; `False` otherwise
        """
        return 'cuda' in self.device


    def save_w(self, path, cnf=None):
        # type: (Union[str, Path], Optional[ConfDict]) -> None
        """
        Save model weights to the specified path.

        :param path: path of the weights file to be saved.
        :param cnf: configuration dictionary (optional) to be saved
            along with the weights.
        """
        torch.save({'state_dict': self.state_dict(), 'cnf': cnf}, path)


    def load_w(self, path):
        # type: (Union[str, Path]) -> None
        """
        Load model weights from the specified path. It also loads the
        configuration dictionary (if available).

        :param path: path of the weights file to be loaded.
        """
        d = torch.load(path, map_location=torch.device(self.device), weights_only=False)
        self.load_state_dict(d['state_dict'])
        self.cnf_dict = d['cnf']
        if d['cnf'] is not None:
            device = d['cnf'].get('device', self.device)
            '''
            il device di cnf è sempre cuda, 
            ma io voglio potrlo eseguire in locale
            aggiungo il seguente controllo
            '''
            if device != self.device:
                d['cnf']['device'] = self.device
                self.to(self.device)
            else:
                self.to(device)


    def requires_grad(self, flag):
        # type: (bool) -> None
        """
        Set the `requires_grad` attribute of all model parameters to `flag`.

        :param flag: True if the model requires gradient, False otherwise.
        """
        for p in self.parameters():
            p.requires_grad = flag
