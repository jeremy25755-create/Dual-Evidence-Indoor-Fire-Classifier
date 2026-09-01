"""Base experiment class and device selection."""

import os

import torch

from models import MODEL_REGISTRY


class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self.model_dict = MODEL_REGISTRY
        if self.args.model not in self.model_dict:
            raise ValueError(
                f"Unknown model {self.args.model!r}. "
                f"Registered models: {', '.join(self.model_dict)}"
            )
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if not self.args.use_gpu:
            print("Use CPU")
            return torch.device("cpu")

        if self.args.gpu == "mps":
            print("Use GPU: MPS")
            return torch.device("mps")

        visible_devices = self.args.devices if self.args.use_multi_gpu else str(self.args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        device = torch.device(f"cuda:{self.args.gpu}")
        print(f"Use GPU: {device}")
        return device

    def _get_data(self):
        raise NotImplementedError

    def vali(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError
