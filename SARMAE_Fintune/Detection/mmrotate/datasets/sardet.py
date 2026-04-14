from mmdet.datasets import CocoDataset 

from .builder import ROTATED_DATASETS
  
@ROTATED_DATASETS.register_module()
class SARDet(CocoDataset):

    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    PALETTE = [(220, 120, 60),(220, 220, 60),(220, 20, 120),(220, 20, 220),(220, 20, 0),(220, 120, 0)]
