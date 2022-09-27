import os
import math
import numpy as np
import torch
from coperception.utils.obj_util import *
from coperception.datasets import V2XSimDet
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from torch.utils.data import Sampler

class MbbSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized
    
    def __init__(self, data_source: Sized, block_len: int) -> None:
        self.data_source = data_source
        self.frame_len = data_source.num_sample_seqs
        self.scene_len = data_source.scene_len
        self.block_len = block_len
        self.frame_pre_scene = self.frame_len // self.scene_len
        self.iter_len = len(self.data_source) // self.block_len * self.block_len

    def __iter__(self) -> Iterator[int]:
        self.iter_list = []
        perm = torch.randperm(len(self.data_source))
        require_range = self.frame_pre_scene - self.block_len
        for idx in perm:
            if idx % self.frame_pre_scene > require_range:
                continue
            if len(self.iter_list) > self.iter_len:
                break
            self.iter_list.extend(range(idx, idx + self.block_len))
        return iter(self.iter_list)

    def __len__(self) -> int:
        return self.iter_len