import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class SpectrogramDataset(Dataset):
    def __init__(self, root_path):
        self.images = sorted([root_path+"/noisy_voice/"+x for x in os.listdir(root_path+"/noisy_voice/")])
        self.targets = sorted([root_path+"/noise/"+x for x in os.listdir(root_path+"/noise/")])

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("L")
        target = Image.open(self.targets[index]).convert("L")

        return self.transform(img), self.transform(target)
    
    def __len__(self):
        return len(self.images)