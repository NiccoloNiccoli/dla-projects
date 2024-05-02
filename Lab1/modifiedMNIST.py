from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import numpy as np

class ModifiedMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, image_size=56):
        super().__init__()
        self.ds = MNIST(root, train, transform, target_transform, download)
        self.transform = transform
        self.image_size = image_size

    def __getitem__(self, index):
        img, target = self.ds[index]
        img2, target2 = self.ds[np.random.randint(0, len(self.ds))]  
        #make a new image and place the original image somewhere in the new image
        new_img = Image.new('1', (self.image_size, self.image_size), 0)
        convert_to_pil = transforms.ToPILImage()
        new_img = self.transform(new_img)
        #sum the two tensors but img is shifted by 16 pixels
        #pick two random numbers between 0 and 32
        x = np.random.randint(0, self.image_size//2 - 28  + 1)
        y = np.random.randint(0, self.image_size - 28)
        #sum two tensors of different shape
        #print(new_img.shape, img.shape)
        new_img[:,x:x+28, y:y+28] = img
        x = np.random.randint(self.image_size//2, self.image_size - 28 + 1)
        y = np.random.randint(0, self.image_size - 28)
        new_img[:,x:x+28, y:y+28] = img2
        # new_img = convert_to_pil(new_img)
        return new_img, target, target2
    
    def __len__(self):
        return len(self.ds)