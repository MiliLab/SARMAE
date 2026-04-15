import torch

def mixed_sar_collate_fn(batch):

    paired_samples = [item for item in batch if item['is_paired']]
    
    result = {}

    all_sar = [item['sar'] for item in batch]
    result['sar'] = torch.stack(all_sar)

    all_sar_target = [item['sar_target'] for item in batch]
    result['sar_target'] = torch.stack(all_sar_target)

    if paired_samples:
        optical_batch = []
        paired_idx = 0
        
        for item in batch:
            if item['is_paired']:
                optical_batch.append(paired_samples[paired_idx]['optical'])
                paired_idx += 1
            else:
                dummy_optical = torch.zeros_like(all_sar[0])
                optical_batch.append(dummy_optical)
        
        result['optical'] = torch.stack(optical_batch)
    else:
        result['optical'] = torch.zeros_like(result['sar'])

    result['is_paired'] = torch.tensor([item['is_paired'] for item in batch])
    
    return result