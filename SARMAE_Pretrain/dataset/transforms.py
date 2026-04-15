from torchvision import transforms

class RepeatChannels:
    def __init__(self, num_channels=3):
        self.num_channels = num_channels
    
    def __call__(self, x):
        return x.repeat(self.num_channels, 1, 1)


class SmartSARTransform:
    def __init__(self, target_channels=3):
        self.target_channels = target_channels
    
    def __call__(self, x):

        if x.mode == 'L':
            original_channels = 1
        elif x.mode == 'RGB':
            original_channels = 3
        elif x.mode == 'RGBA':
            original_channels = 4
        else:
            x = x.convert('RGB')
            original_channels = 3

        transform_to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        img_tensor = transform_to_tensor(x)

        if original_channels == 1:
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(self.target_channels, 1, 1)
        elif original_channels == 3:
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor[:self.target_channels]
        elif original_channels == 4:
            if img_tensor.shape[0] == 4:
                img_tensor = img_tensor[:self.target_channels]
        else:
            x = x.convert('RGB')
            img_tensor = transform_to_tensor(x)
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor[:self.target_channels]
        
        return img_tensor

def build_sar_transform(size=224):
    return transforms.Compose([
        SmartSARTransform(target_channels=3),  
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def build_optical_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])