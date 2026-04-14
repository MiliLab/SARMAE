from mmdet.datasets import CocoDataset 

from .builder import ROTATED_DATASETS
  
@ROTATED_DATASETS.register_module()
class SSDDDataset(CocoDataset):

    CLASSES = ('ship',)

    PALETTE = [[0, 255, 0]]