import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import utils
from base import BaseDataLoader
import os, json
import pandas as pd
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
from utils import util
from sklearn.model_selection import StratifiedShuffleSplit


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

        self.dataset = TinyImageNetDatasets(path=self.data_dir, train=training, transform=trsfm, split=validation_split,
                                            nums=augment_pics, flag=run_from_image_and_save_tensor)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers,
                         assigned_val=assign_val_sample, samplers=self.dataset.samples)


class TinyImageNetDatasets(Dataset):
    def __init__(self, path, train, transform, split, nums, flag):
        super().__init__()

        image_dir = os.path.join(path, "images")
        assert os.path.exists(image_dir), "dir:'{}' not found.".format(image_dir)

        self.target_transform = None
        self.image_folder = image_dir
        self.transform = transform
        self.root = path

        csv_mode = 'train' if train else 'test'
        if flag:
            self.run_from_image(train, path, split, nums, image_dir, transform)
            utils.save_as_pickle(os.path.join(path, csv_mode + '_data.pickle'), self.data)
            utils.save_as_pickle(os.path.join(path, csv_mode + '_targets.pickle'), self.targets)
            if train:  # 测试集没有这个变量
                utils.save_as_pickle(os.path.join(path, 'samples.pickle'), self.samples)
        else:
            self.data = utils.load_from_pickle(os.path.join(path, csv_mode + '_data.pickle'))
            self.targets = utils.load_from_pickle(os.path.join(path, csv_mode + '_targets.pickle'))
            if train:  # 测试集没有这个变量
                self.samples = utils.load_from_pickle(os.path.join(path, 'samples.pickle'))

    def run_from_image(self, train, path, split, nums, image_dir, transform):
        # 根据布尔值train，来确定是生成训练集数据（如果是训练集，那么肯定也要生成验证集），还是测试集数据
        csv_mode = 'train' if train else 'test'
        csv_name = csv_mode + '.csv'
        csv_path = os.path.join(path, csv_name)
        assert os.path.exists(csv_path), "file:'{}' not found. please run creat_tiny.py firstly".format(csv_path)
        csv_data = pd.read_csv(csv_path)

        csv_image_name_list = list(csv_data['filename'])
        csv_image_label_list = list(set(csv_data['label']))  # set具有唯一性，但是无序
        csv_image_label_list.sort()  # 注意list的sort函数为就地sort

        # 生成self.target
        for idx, image_label in enumerate(csv_image_label_list):
            csv_data.replace(image_label, idx, inplace=True)
        self.targets = torch.from_numpy(np.array(csv_data['label'])).long()

        if train:
            self.samples = self.cal_samples(split, self.targets)
            csv_train_image_name_list = csv_data['filename'][self.samples[0].indices]
            csv_valid_image_name_list = csv_data['filename'][self.samples[1].indices]
            train_val_size = len(csv_train_image_name_list) + len(csv_valid_image_name_list)
            train_size = len(csv_train_image_name_list)
            data = torch.empty((train_val_size, 3, 224, 224))

            additional_data = torch.empty((nums * train_size, 3, 224, 224))  # 用以记录数据增强的额外tenor
            additional_target = torch.empty((nums * train_size))  # 用以记录数据增强的额外tensor的标签
            addidional_idx = 0  # 当前的额外标签

            # 处理训练集
            for idx, image_name in enumerate(csv_train_image_name_list):
                image_path = os.path.join(os.getcwd(), image_dir, image_name)
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                temp = transform['train'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）
                data[self.samples[0].indices[idx]] = temp

                # 添加额外图片
                for _ in range(nums):
                    temp = transform['train'](image)  # 由于采用了transforms.RandAugment()，本来就有随机性
                    additional_data[addidional_idx, :] = temp
                    additional_target[addidional_idx] = self.targets[self.samples[0].indices[idx]]
                    addidional_idx = addidional_idx + 1

                if (idx + 1) % 100 == 0:
                    print(f'There are {train_val_size} images. Processing image {idx + 1}')

            # 拼接额外图片
            temp = np.array([i + train_size for i in range(addidional_idx)], dtype=np.int64)  # 索引只能为int，默认是float
            self.samples[0].indices = np.hstack((self.samples[0].indices, temp))
            data = torch.cat([data, additional_data], dim=0)
            self.targets = torch.cat([self.targets, additional_target], dim=0).long()

            # 处理验证集
            for idx, image_name in enumerate(csv_valid_image_name_list):
                image_path = os.path.join(os.getcwd(), image_dir, image_name)
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                image = transform['val'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）

                data[self.samples[1].indices[idx]] = image

                if (train_size + idx + 1) % 100 == 0:
                    print(f'There are{train_val_size} images. Processing image {train_size + idx + 1}')

        else:
            data = torch.empty((len(csv_image_name_list), 3, 224, 224))
            for idx, image_name in enumerate(csv_image_name_list):
                image_path = os.path.join(os.getcwd(), image_dir, image_name)
                # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
                image = Image.open(image_path).convert('RGB')
                image = transform['test'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）

                data[idx] = image

        self.data = data

        """
        #之后的任务：探究不同数据增强对图片的影响，对训练结果的影响，并对pytorch-template进行修改，添加事先分隔号训练集、验证集的功能
        data_list = []
        for idx, image_name in enumerate(csv_image_name_list):
            image_path = os.path.join(os.getcwd(), image_dir, image_name)
            # 读出来的图像是RGBA四通道的，A通道为透明通道，该通道值对深度学习模型训练来说暂时用不到，因此使用convert(‘RGB’)进行通道转换
            image = Image.open(image_path).convert('RGB')
            image = transform['train'](image)  # .unsqueeze(0)增加维度（0表示，在第一个位置增加维度）

            data_list.append(image)

            ############################
            # temp_trsfm = transforms.Compose([transforms.Resize(224)
            #                                  transforms.RandomHorizontalFlip(),
            #                                  transforms.ToTensor()])
            # image_path = os.path.join(os.getcwd(), image_dir, image_name)
            # image = Image.open(image_path).convert('RGB')
            # image = temp_trsfm(image)
            # transforms.ToPILImage()(image).show()

        self.data = torch.stack(data_list)
        del data_list
        """

        num_label2text_label = dict([(idx, text) for idx, text in enumerate(csv_image_label_list)])
        # util.write_json(num_label2text_label, os.path.join(path, 'num_label2text_label.json'))

    def cal_samples(self, rate, targets):
        """事先分割好训练集与验证集，方便后续不同的transform"""
        split = StratifiedShuffleSplit(n_splits=1, test_size=rate, random_state=42)
        for train_index, valid_index in split.split(np.zeros(len(targets)), np.array(targets)):
            train_sampler, valid_sampler = SubsetRandomSampler(train_index), SubsetRandomSampler(valid_index)
        return [train_sampler, valid_sampler]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        return self.data[item, :], self.targets[item]