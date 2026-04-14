# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .dota import DOTADataset
import glob
import os.path as osp
import os
import mmcv
import numpy as np
import torch

@ROTATED_DATASETS.register_module()
class RSARDataset(DOTADataset):
    """RSAR Dataset for rotated object detection.
    
    RSAR dataset contains mixed image formats (.jpg and .png).
    This class extends DOTADataset to handle mixed formats automatically.
    """
    
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
    PALETTE = [(220, 120, 60),(220, 220, 60),(220, 20, 120),(220, 20, 220),(220, 20, 0),(220, 120, 0)]
    
    def load_annotations(self, ann_folder):
        """Load annotations with auto-detection of image format (.jpg, .png, or .bmp).
        
        Args:
            ann_folder: folder that contains RSAR annotations txt files
            
        Returns:
            list[dict]: Annotation information with corrected image filenames
        """
        # Call parent's load_annotations (assumes .png by default)
        data_infos_temp = super().load_annotations(ann_folder)
        
        # Fix the filename by auto-detecting actual image format
        data_infos = []
        skipped_count = 0
        format_count = {'jpg': 0, 'png': 0, 'bmp': 0}
        
        for data_info in data_infos_temp:
            old_filename = data_info['filename']
            img_id = old_filename[:-4]  # Remove extension
            
            # Check which format actually exists (.jpg, .png, or .bmp)
            img_path_jpg = osp.join(self.img_prefix, img_id + '.jpg')
            img_path_png = osp.join(self.img_prefix, img_id + '.png')
            img_path_bmp = osp.join(self.img_prefix, img_id + '.bmp')
            
            if osp.exists(img_path_jpg):
                data_info['filename'] = img_id + '.jpg'
                data_infos.append(data_info)
                format_count['jpg'] += 1
            elif osp.exists(img_path_png):
                # Keep .png (already set by parent class)
                data_infos.append(data_info)
                format_count['png'] += 1
            elif osp.exists(img_path_bmp):
                data_info['filename'] = img_id + '.bmp'
                data_infos.append(data_info)
                format_count['bmp'] += 1
            else:
                skipped_count += 1
                continue
        
        # Print summary
        print(f"RSAR Dataset: Loaded {len(data_infos)} samples "
              f"(jpg: {format_count['jpg']}, png: {format_count['png']}, bmp: {format_count['bmp']})")
        if skipped_count > 0:
            print(f"  Warning: Skipped {skipped_count} annotation files without corresponding images")
        
        return data_infos
