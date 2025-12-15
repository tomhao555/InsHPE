import argparse
import os

import random
import progressbar
import time
import logging
import pdb
from tqdm import tqdm
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from model.transfer_hand import define_G
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data import DataLoader, Subset

import random
from typing import Any

import os
os.environ['TMPDIR'] = 'your_path/tmp'



parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python train.py

parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate at t=0')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (SGD only)')

parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay (SGD only)')
parser.add_argument('--learning_rate_decay', type=float, default=1e-7, help='learning rate decay')

parser.add_argument('--JOINT_NUM', type=int, default = 23,  help='number of joints')
parser.add_argument('--stacks', type=int, default = 500, help='start epoch')
parser.add_argument('--start_epoch', type=int, default = 0, help='start epoch')

parser.add_argument('--save_root_dir', type=str, default='results',  help='output folder')
parser.add_argument('--model', type=str, default = '', help='model name for training resume')
parser.add_argument('--optimizer', type=str, default = '', help='optimizer name for training resume')

parser.add_argument('--dataset', type=str, default = 'dexycb', help='dataset name: nyu, dexycb..')
parser.add_argument('--dataset_path', type=str, default = '../dataset',  help='dataset path')
parser.add_argument('--protocal', type=str, default = 's0',  help='evaluation setting')

parser.add_argument('--test_path', type=str, default = '../dataset',  help='model name for training resume')

parser.add_argument('--model_name', type=str, default = 'handdagt_distillation',  help='')
parser.add_argument('--gpu', type=str, default = '0',  help='gpu')
parser.add_argument('--add_info', type=str, default = 'info')

parser.add_argument('--inc_ratio', type=float, default = 1.0, help='inc_train_data')
parser.add_argument('--inc_sample_ratio', type=float, default = 0.2, help='sample_train_data')
parser.add_argument('--nyu_sample_ratio', type=float, default = 0.8, help='sample_train_data')
parser.add_argument('--all_sample_ratio', type=float, default = 0.1, help='sample_train_data')

parser.add_argument('--step_size', type=float, default = 80, help='sample_size')
parser.add_argument('--backbone_pth', type=str, default = '', help='teacher_backbone_name')

opt = parser.parse_args()

module = importlib.import_module('network_'+opt.model_name)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

torch.cuda.set_device(opt.main_gpu)

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


class ConcatDataset(Dataset):
	"""
    将多个数据集线性拼接在一起，支持标准索引访问。

    该实现避免了原版本中可能导致索引计算错误的逻辑，采用更直观的二分查找方式定位样本位置。
    """

	def __init__(self, *datasets: Dataset):
		self.datasets = datasets
		self.cumulative_sizes = self._compute_cumulative_sizes()

	def _compute_cumulative_sizes(self) -> list:
		"""计算各数据集累计长度"""
		sizes = [len(d) for d in self.datasets]
		cum_sizes = [0]
		current_sum = 0
		for size in sizes:
			current_sum += size
			cum_sizes.append(current_sum)
		return cum_sizes

	def __getitem__(self, idx: int) -> Any:
		"""
        根据全局索引找到对应的数据集和局部索引

        参数:
            idx (int): 全局索引

        返回:
            对应的样本
        """
		# 使用二分查找确定样本所在子数据集
		left, right = 0, len(self.cumulative_sizes) - 1
		while left < right:
			mid = (left + right) // 2
			if idx >= self.cumulative_sizes[mid]:
				left = mid + 1
			else:
				right = mid

		dataset_idx = left - 1
		local_idx = idx - self.cumulative_sizes[dataset_idx]
		return self.datasets[dataset_idx][local_idx]

	def __len__(self) -> int:
		"""
        返回总样本数

        返回:
            int: 所有数据集样本总数
        """
		return self.cumulative_sizes[-1]


def sample_dataset(dataset, ratio):
	"""
	从 dataset 中随机采样 ratio 比例的数据
	"""
	num_samples = int(len(dataset) * ratio)
	indices = torch.randperm(len(dataset))[:num_samples]
	return Subset(dataset, indices)


class transfer_img(object):
	def __init__(self, model_path):
		torch.cuda.set_device(0)
		# load ori transfer net
		self.transferNet = define_G(1, 1, 64, 'resnet_9blocks', 'instance', False, 'xavier').cuda()
		self.transferNet.requires_grad_(False)
		self.transferNet.eval()
		if model_path!= '':
			model_dict = torch.load(model_path + '/latest_net_G_A.pth',
									map_location=lambda storage, loc: storage)
			self.transferNet.load_state_dict(model_dict)

	def transfer_syn(self, img):
		transfered_img = self.transferNet(img)

		"""
                注入相机噪声和离群点，模拟传感器噪声
                参数:
                    depth_img: 输入深度图 (tensor, shape: [B, C, H, W])
                    offset_range: 深度值随机偏移范围（±mm）
                    outlier_ratio: 离群点比例（设为背景深度的像素比例）
                    bg_depth: 背景深度值
                返回:
                    noisy_depth: 加噪后的深度图 (tensor, shape: [B, C, H, W])
                """
		depth_img = transfered_img
		batch_size, channels, h, w = depth_img.shape

		offset_range = 0.005
		outlier_ratio = 0.08
		bg_depth = 1.0

		# 1. 注入深度值随机偏移（高斯噪声）
		noise = torch.randn_like(depth_img) * offset_range
		noisy_depth = depth_img + noise
		noisy_depth = torch.clamp(noisy_depth, 0.0, bg_depth)

		# 2. 注入离群点（设为背景深度）
		num_outliers = int(h * w * outlier_ratio)
		for b in range(batch_size):
			# 随机选一些点作为离群点
			outlier_indices = torch.randperm(h * w, device=depth_img.device)[:num_outliers]
			outlier_y = outlier_indices // w
			outlier_x = outlier_indices % w
			noisy_depth[b, :, outlier_y, outlier_x] = bg_depth

		return transfered_img


if opt.dataset == 'dexycb':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_'+opt.protocal +'_' + opt.model_name+'_'+str(opt.stacks)+'stacks')
	from dataloader import loader
	opt.JOINT_NUM = 21
elif opt.dataset == 'ho3d':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_v2_' + opt.model_name+'_'+ str(opt.stacks)+'stacks')
	from dataloader import ho3d_loader
elif opt.dataset == 'nyu':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+ str(opt.stacks)+'stacks' + '_'  + opt.add_info)
	from dataloader import loader
	opt.JOINT_NUM = 23
elif 'IncNYU' in opt.dataset:
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+ str(opt.stacks)+'stacks'+'_'+ opt.add_info)
	from dataloader import loader
	opt.JOINT_NUM = 23


calculate = [0, 2,
             4, 6,
             8, 10,
             12, 14,
             16, 17, 18,
             20, 21, 22]
def _debug(model):
	model = model.netR_1
	print(model.named_paramters())
try:
	os.makedirs(save_dir)
except OSError:
	pass

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
					filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')

sampler = None

# 1. Load data
if opt.dataset == 'nyu':
	data_dir = 'your_path/datasets/nyu'
	train_data = loader.nyu_loader(data_dir, 'train', aug_para=[10, 0.2, 180],
										  joint_num=opt.JOINT_NUM, center_type='refine')
elif opt.dataset == 'ho3d':
	train_data = ho3d_loader.HO3D('train_all', opt.dataset_path, aug_para=[10, 0.2, 180], dataset_version='v2', center_type='joint_mean' )
elif opt.dataset == 'dexycb' :
	train_data = loader.DexYCBDataset(opt.protocal, 'train', opt.dataset_path, aug_para=[10, 0.2, 180])
elif 'IncNYU' in opt.dataset:
	teacher_data_dir = 'your_path/datasets/nyu'
	train_data_inc = loader.IncNYU_loader(opt.dataset_path, teacher_data_dir, 'train', aug_para=[10, 0.2, 180], joint_num=opt.JOINT_NUM, center_type='refine', flag=True)
	train_data_nyu = loader.IncNYU_loader(opt.dataset_path, teacher_data_dir, 'train', aug_para=[10, 0.2, 180], joint_num=opt.JOINT_NUM, center_type='refine')

	train_data_inc_sampled = sample_dataset(train_data_inc, opt.inc_sample_ratio)
	train_data_nyu_sampled = sample_dataset(train_data_nyu, opt.nyu_sample_ratio)

	train_data = ConcatDataset(train_data_inc_sampled, train_data_nyu_sampled)

	train_data = sample_dataset(train_data, opt.all_sample_ratio)



train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batchSize,
													shuffle=True, num_workers=int(opt.workers), pin_memory=False)



if opt.dataset == 'dexycb' :
	test_data = loader.DexYCBDataset(opt.protocal, 'test', opt.dataset_path)
elif opt.dataset == 'ho3d':
	test_data = ho3d_loader.HO3D('test', opt.dataset_path, dataset_version='v2', center_type='joint_mean' )
elif opt.dataset == 'nyu':
	test_data = loader.nyu_loader(opt.dataset_path, 'test', joint_num=opt.JOINT_NUM)
	test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)
elif 'IncNYU' in opt.dataset :
	teacher_data_dir = 'your_path/datasets/nyu'
	test_data_inc = loader.IncNYU_loader(opt.dataset_path, teacher_data_dir, 'test', aug_para=[10, 0.2, 180],
										  joint_num=opt.JOINT_NUM, center_type='refine', flag=True)
	test_data_nyu = loader.IncNYU_loader(opt.dataset_path, teacher_data_dir, 'test', aug_para=[10, 0.2, 180],
										  joint_num=opt.JOINT_NUM, center_type='refine')

	test_dataloader_inc = torch.utils.data.DataLoader(test_data_inc, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)
	test_dataloader_nyu = torch.utils.data.DataLoader(test_data_nyu, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

print(opt)

# 2. Define model, loss and optimizer
backbone_dir = 'your_path/projects/HPE/HandDAGT_distillation/HandDAGT-main/pretrained_model'
teacher_backbone_pth = os.path.join(backbone_dir, opt.backbone_pth)


model = getattr(module, 'HandModel')(joints=opt.JOINT_NUM, stacks=opt.stacks, teacher_model_path=teacher_backbone_pth)

# 统计模型中的总参数量
total_params = sum(p.numel() for p in model.parameters())

# 统计可训练参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

if opt.ngpu > 1:
	model.netR_1 = torch.nn.DataParallel(model.netR_1, range(opt.ngpu))
	model.netR_2 = torch.nn.DataParallel(model.netR_2, range(opt.ngpu))
	model.netR_3 = torch.nn.DataParallel(model.netR_3, range(opt.ngpu))
if opt.model != '':
	model.load_state_dict(torch.load(os.path.join(save_dir, opt.model)))

model.cuda()

parameters = model.parameters()

optimizer = optim.AdamW(parameters, lr=opt.learning_rate, betas = (0.5, 0.999), eps=1e-06, weight_decay=opt.weight_decay)
if opt.optimizer != '':
	optimizer.load_state_dict(torch.load(os.path.join(save_dir, opt.optimizer)))

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=0.1)

if opt.dataset == 'dexycb':
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
if opt.dataset == 'ho3d':
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

test_best_error_mix = np.inf

test_best_error_com = np.inf

test_best_error_inc = np.inf



# 3. Training and testing
for epoch in range(opt.start_epoch, opt.nepoch):
	scheduler.step(epoch)
	if opt.dataset == 'msra':
		print('======>>>>> Online epoch: #%d, Test: %s, lr=%f  <<<<<======' % (
		epoch, subject_names[opt.test_index], scheduler.get_lr()[0]))
	else:
		print('======>>>>> Online epoch: #%d, lr=%f  <<<<<======' % (epoch, scheduler.get_lr()[0]))

	# 3.1 switch to train mode
	torch.cuda.synchronize()
	model.train()
	train_mse = 0.0
	train_mse_wld = 0.0
	timer = time.time()

	for i, data in enumerate(tqdm(train_dataloader, ncols=50)):

		if len(data[0]) == 1:
			continue
		torch.cuda.synchronize()
		# 3.1.1 load inputs and targets

		if opt.dataset == "nyu":
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length, \
				teacher_data, teacher_pcl, teacher_center, teacher_M, teacher_cube = data
			volume_length = volume_length.cuda()
			teacher_data = teacher_data.cuda()
			teacher_pcl = teacher_pcl.cuda()
			teacher_center = teacher_center.cuda()
			teacher_M = teacher_M.cuda()
			teacher_cube = teacher_cube.cuda()
		elif "IncNYU" in opt.dataset:
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length, \
			teacher_data, teacher_pcl, teacher_center, teacher_M, teacher_cube = data
			volume_length = volume_length.cuda()
			teacher_data = teacher_data.cuda()
			teacher_pcl = teacher_pcl.cuda()
			teacher_center = teacher_center.cuda()
			teacher_M = teacher_M.cuda()
			teacher_cube = teacher_cube.cuda()
		else:
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para = data
		points, gt_xyz, img = points.cuda(), gt_xyz.cuda(), img.cuda()
		center, M, cube, cam_para = center.cuda(), M.cuda(), cube.cuda(), cam_para.cuda()

		# 风格迁移IncNYU
		if ("IncNYU" in opt.dataset):
			hand_depth = img.float()
			transfer = transfer_img(".")
			transfered_hand_depth = transfer.transfer_syn(hand_depth)

			img = transfered_hand_depth

		# 3.1.2 compute output
		optimizer.zero_grad()
		loss = model.get_loss(points.transpose(1, 2), points.transpose(1, 2), img, train_data, center, M, cube, cam_para, gt_xyz.transpose(1, 2),
							  teacher_pcl.transpose(1, 2), teacher_data, teacher_center, teacher_M, teacher_cube)

		# 3.1.3 compute gradient and do SGD step
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		torch.cuda.synchronize()

		# 3.1.4 update training error
		train_mse = train_mse + loss.item() * len(points)

	# time taken
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(train_data)
	print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

	# print mse
	train_mse = train_mse / len(train_data)

	print('mean-square error of 1 sample: %f, #train_data = %d' % (train_mse, len(train_data)))

	if (epoch % 10) == 0:
		torch.save(model.state_dict(), '%s/netR_%d.pth' % (save_dir, epoch))
		torch.save(optimizer.state_dict(), '%s/optimizer_%d.pth' % (save_dir, epoch))
		

	# 3.2 switch to evaluate mode
	torch.cuda.synchronize()
	model.eval()
	timer = time.time()

	# 3.2.0 保存在nyu测试集上表现最好的模型
	test_mse = 0.0
	test_wld_err = 0.0
	for i, data in enumerate(tqdm(test_dataloader, ncols=50)):
		torch.cuda.synchronize()
		with torch.no_grad():
			# 3.2.1 load inputs and targets

			if opt.dataset == "nyu":
				img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length = data
				volume_length = volume_length.cuda()
			elif "IncNYU" in opt.dataset:
				img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length, \
				teacher_data, teacher_pcl, teacher_center, teacher_M, teacher_cube = data
				volume_length = volume_length.cuda()
				teacher_data = teacher_data.cuda()
				teacher_pcl = teacher_pcl.cuda()
				teacher_center = teacher_center.cuda()
				teacher_M = teacher_M.cuda()
				teacher_cube = teacher_cube.cuda()
			else:
				img, points, gt_xyz, uvd_gt, center, M, cube, cam_para = data
				volume_length = 250.
			points, gt_xyz, img = points.cuda(), gt_xyz.cuda(), img.cuda()
			center, M, cube, cam_para = center.cuda(), M.cuda(), cube.cuda(), cam_para.cuda()

			# 风格迁移
			if "IncNYU" in opt.dataset:
				hand_depth = img.float()
				transfer = transfer_img(".")
				transfered_hand_depth = transfer.transfer_syn(hand_depth)

				img = transfered_hand_depth

			estimation = model(points.transpose(1, 2), points.transpose(1, 2), img, test_data, center, M, cube,
							   cam_para)

		torch.cuda.synchronize()

		# 3.2.3 compute error in world cs
		outputs_xyz = estimation.transpose(1, 2)
		diff = torch.pow(outputs_xyz - gt_xyz, 2).view(-1, opt.JOINT_NUM, 3)
		diff_sum = torch.sum(diff, 2)
		diff_sum_sqrt = torch.sqrt(diff_sum)
		if opt.JOINT_NUM == 23:
			diff_sum_sqrt = diff_sum_sqrt[:, calculate]
		diff_mean = torch.mean(diff_sum_sqrt, 1).view(-1, 1)
		diff_mean_wld = torch.mul(diff_mean, volume_length.view(-1, 1) / 2)
		test_wld_err = test_wld_err + diff_mean_wld.sum().item()

	if test_best_error_com > test_wld_err:
		test_best_error_com = test_wld_err
		torch.save(model.state_dict(), '%s/nyu_refine_nyu_joint_best_model_nyu.pth' % (save_dir))
		torch.save(optimizer.state_dict(), '%s/best_optimizer_nyu.pth' % (save_dir))




	# time taken
	test_data_inc = test_data
	test_best_error_inc = test_best_error_com
	torch.cuda.synchronize()
	timer = time.time() - timer
	timer = timer / len(test_data_inc) / 2
	print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
	# print mse
	test_mse = test_mse / len(test_data_inc)
	print('mean-square error of 1 sample: %f, #test_data = %d' % (test_mse, len(test_data_inc)))
	test_wld_err = test_wld_err / len(test_data_inc)
	print('average estimation error in world coordinate system: %f (mm)' % (test_wld_err))
	# log
	logging.info(
		'Epoch#%d: train error=%e, train wld error = %f mm, test error=%e, test wld error = %f mm, best wld error = %f' % (
		epoch, train_mse, train_mse_wld, test_mse, test_wld_err, test_best_error_inc / len(test_data_inc)))
