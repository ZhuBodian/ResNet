from torchvision import datasets, transforms
from base import BaseDataLoader
from base import base_my_dataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 assign_val_sample=False, augment_pics=0):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         assigned_val=assign_val_sample, samplers=None)


class TinyImageNetDataloader(BaseDataLoader):
    """
    MiniImageNet data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 assign_val_sample=False, augment_pics=0):
        trsfm = {
            "train": transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.RandAugment(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize([224, 224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor()])
        }
        self.data_dir = data_dir

        # 一般第一次运行是true（从图像中读取数据并保存为pickle），第二次为false（直接从pickle中读数，省去预处理时间，且不用再存）
        run_from_image_and_save_tensor = False

        self.dataset = base_my_dataset.BaseMyDataset(path=self.data_dir, train=training, transform=trsfm, split=validation_split,
                                            nums=augment_pics, flag=run_from_image_and_save_tensor)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         assigned_val=assign_val_sample, samplers=self.dataset.samples)


class Caltech101Dataloader(BaseDataLoader):
    """
    MiniImageNet data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 assign_val_sample=False, augment_pics=0):
        trsfm = {
            "train": transforms.Compose([transforms.Resize([224, 224]),
                                         transforms.RandAugment(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([transforms.Resize([224, 224]),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "test": transforms.Compose([transforms.Resize([224, 224]),
                                        transforms.ToTensor()])
        }
        self.data_dir = data_dir

        # 一般第一次运行是true（从图像中读取数据并保存为pickle），第二次为false（直接从pickle中读数，省去预处理时间，且不用再存）
        run_from_image_and_save_tensor = True

        self.dataset = base_my_dataset.BaseMyDataset(path=self.data_dir, train=training, transform=trsfm, split=validation_split,
                                            nums=augment_pics, flag=run_from_image_and_save_tensor)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         assigned_val=assign_val_sample, samplers=self.dataset.samples)


