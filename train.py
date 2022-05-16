import argparse
import collections
import os.path
import torch.nn as nn
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import utils
from utils import send_email
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from utils import global_var

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    global_var._init()
    global_var.set_value('email_log', send_email.Mylog(header='pytorch_template', subject='测试log'))

    # config为parse_config中的ConfigParser类，且注意config.init_obj第二个变量均为某个模块，且变量在__name__ == '__main__'中的special variables中
    # 第一个变量为str类型，详细查看config.json文件
    logger = config.get_logger('train')  # 创建指定名字的日志文件，返回的为标准logging类

    # setup data_loader instances，实例化data_loader.data_loaders.MnistDataLoader（config.json中data_loader的type为这个）
    data_loader = config.init_obj('data_loader', module_data)  # 返回一个名为data_loader的，module_data实例
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console，并读取预训练权重
    model = module_arch.resnet34(num_classes=1000)  # 如果直接设置为10，后面无法导入预训练权重
    if config.config['arch']['load_pretrain_weights']:
        path = config.config['arch']['pretrain_weights_name']
        assert os.path.exists(path), f'file {path} not exists'
        device, device_ids = prepare_device(config['n_gpu'])
        model.load_state_dict(torch.load(path, map_location=device))
        in_channel = model.fc.in_features
        model.fc = nn.Linear(in_channel, 10)  # 10为自建数据集的类别

    logger.info(model)  # logging.info('输出信息')，而类model的输出信息为可训练参数
    global_var.get_value('email_log').add_log(model.__str__())

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # config['loss']为'nll_loss'；criterion返回的实际是module_loss模块下的nll_loss函数的函数句柄，在special variables里面；
    criterion = getattr(module_loss, config['loss'])  # 损失函数的函数句柄
    metrics = [getattr(module_metric, met) for met in config['metrics']]  # tensorboard scalars中要跟踪的数据

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())  # 一个可迭代对象
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    global_var.get_value('email_log').send_mail()
    os.system('shutdown')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    # 这里专门把需要多次更改的超参数给列了出来，比如json中batch_size为为128，要想改为256，可以python train.py -c config.json --bs 256
    # 改epoch是我自己加的（方便快速调试），将原来的100改为5
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')  # config.json之外的，添加到args中的参数
    options = [
        # target是该超参数在config.json中的层级结构。如果要
        # CustomArgs(flags=['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        # CustomArgs(flags=['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(flags=['--ep', '--epochs'], type=int, target='trainer;epochs')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
