import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import FashionMNIST

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class FashionMNIST_truncated(data.Dataset):
    """截断版Fashion-MNIST数据集，允许基于索引选择子集"""

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        fashion_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # 获取训练数据
            data = fashion_dataobj.data.numpy()
            target = np.array(fashion_dataobj.targets)
        else:
            # 获取测试数据
            data = fashion_dataobj.data.numpy()
            target = np.array(fashion_dataobj.targets)

        # 如果指定了索引，则只返回相应索引的数据
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # 将NumPy数组转换为PIL图像，因为大多数转换操作都需要PIL图像
        img = Image.fromarray(img.astype(np.uint8), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)