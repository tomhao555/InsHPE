'''
evaluation
'''
import argparse

import os
import random
import time
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
from util import vis_tool
import shutil
import logging
import cv2


from datetime import datetime  # 从模块导入同名类
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
parser.add_argument('--nepoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--ngpu', type=int, default=1, help='# GPUs')
parser.add_argument('--main_gpu', type=int, default=0, help='main GPU id') # CUDA_VISIBLE_DEVICES=0 python eval.py

parser.add_argument('--size', type=str, default='full', help='how many samples do we load: small | full')
parser.add_argument('--bit_width', type=int, default=4, help='quantize for bit width')
parser.add_argument('--SAMPLE_NUM', type=int, default = 1024,  help='number of sample points')
parser.add_argument('--JOINT_NUM', type=int, default = 21,  help='number of joints')
parser.add_argument('--INPUT_FEATURE_NUM', type=int, default = 3,  help='number of input point features')
parser.add_argument('--stacks', type=int, default = 3, help='start epoch')

parser.add_argument('--save_root_dir', type=str, default='./results',  help='output folder')
parser.add_argument('--model', type=str, default = 'nyu_refine_nyu_joint_best_model_nyu.pth',  help='model name for training resume')
parser.add_argument('--test_path', type=str, default = '../dataset',  help='model name for training resume')
parser.add_argument('--protocal', type=str, default = 's0',  help='model name for training resume')

parser.add_argument('--dataset', type=str, default = 'dexycb', help='optimizer name for training resume')
parser.add_argument('--model_name', type=str, default = 'handdagt_tgdnet',  help='')
parser.add_argument('--gpu', type=str, default = '3',  help='gpu')

parser.add_argument('--add_info', type=str, default = 'info')

parser.add_argument('--model_checkpoint_dir', type=str, default = '',  help='')

opt = parser.parse_args()
# print (opt)

module = importlib.import_module('network_'+opt.model_name)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

opt.manualSeed = 1
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

def convert_nyu2mano(joint):
	select_joint = joint.clone()
	select_joint[:, 1, :] = joint[:, 1, :] + (joint[:, 2, :] - joint[:, 1, :]) * 0.3
	select_joint[:, 5, :] = joint[:, 5, :] + (joint[:, 6, :] - joint[:, 5, :]) * 0.3
	select_joint[:, 9, :] = joint[:, 9, :] + (joint[:, 10, :] - joint[:, 9, :]) * 0.3
	select_joint[:, 13, :] = joint[:, 13, :] + (joint[:, 14, :] - joint[:, 13, :]) * 0.3
	select_joint[:, 17, :] = joint[:, 17, :] + (joint[:, 18, :] - joint[:, 17, :]) * 0.2

	select_joint[:, 0, :] = joint[:, 0, :] - (joint[:, 1, :] - joint[:, 0, :]) * 0.3
	select_joint[:, 4, :] = joint[:, 4, :] - (joint[:, 5, :] - joint[:, 4, :]) * 0.3
	select_joint[:, 8, :] = joint[:, 8, :] - (joint[:, 9, :] - joint[:, 8, :]) * 0.3
	select_joint[:, 12, :] = joint[:, 12, :] - (joint[:, 13, :] - joint[:, 12, :]) * 0.3
	select_joint[:, 16, :] = joint[:, 16, :] - (joint[:, 17, :] - joint[:, 16, :]) * 0.3

	select_joint[:, 3, :] = joint[:, 3, :] - (joint[:, 3, :] - joint[:, 2, :]) * 0.1
	select_joint[:, 7, :] = joint[:, 7, :] - (joint[:, 7, :] - joint[:, 6, :]) * 0.1
	select_joint[:, 11, :] = joint[:, 11, :] - (joint[:, 11, :] - joint[:, 10, :]) * 0.2
	select_joint[:, 15, :] = joint[:, 15, :] - (joint[:, 15, :] - joint[:, 14, :]) * 0.3

	NYU2MANO = [22,
				15,14,13,
				11,10,9,
				3,2,1,
				7,6,5,
				19,18,17,
				12,8,0,4,16]
	return select_joint[:, NYU2MANO, :]

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

		return img


if opt.dataset == 'dexycb':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_'+opt.protocal +'_' + opt.model_name+'_'+str(opt.stacks)+'stacks')
	from dataloader import loader 
	opt.JOINT_NUM = 21
elif opt.dataset == 'nyu':
	save_dir = os.path.join(opt.save_root_dir, opt.dataset+ '_' + opt.model_name+'_'+ str(opt.stacks)+'stacks')
	from dataloader import loader 
	opt.JOINT_NUM = 23
	# calculate = [0, 2,
	# 			 4, 6,
	# 			 8, 10,
	# 			 12, 14,
	# 			 16, 17, 18,
	# 			 20, 21, 22]
elif 'IncNYU' in opt.dataset:
	save_dir = os.path.join(opt.save_root_dir, opt.dataset + '_' + opt.model_name + '_' + str(opt.stacks) + 'stacks')
	from dataloader import loader
	# opt.JOINT_NUM = 21
	opt.JOINT_NUM = 23

#
# calculate = [0, 2,
# 			 4, 6,
# 			 8, 10,
# 			 12, 14,
# 			 16, 17, 18,
# 			 21, 22, 20]

calculate = [0, 2,
			 4, 6,
			 8, 10,
			 12, 14,
			 16, 17, 18,
			 20, 21, 22]

# 1. Load data                                         
if opt.dataset == 'dexycb' :
	test_data = loader.DexYCBDataset(opt.protocal, 'test', opt.test_path)
elif opt.dataset == 'nyu':
	test_data = loader.nyu_loader(opt.test_path, 'test', joint_num=opt.JOINT_NUM)
elif 'IncNYU' in opt.dataset :
	test_data = loader.IncNYU_pure_loader(opt.test_path, 'test', joint_num=opt.JOINT_NUM)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batchSize,
										  shuffle=False, num_workers=int(opt.workers), pin_memory=False)

opt.model_dir = os.path.join(save_dir, 'vis')
opt.model_dir += opt.add_info
if not os.path.exists(opt.model_dir):
	os.makedirs(opt.model_dir)
	os.makedirs(opt.model_dir + '/img')
	os.makedirs(opt.model_dir + '/debug')
	os.makedirs(opt.model_dir + '/files')
# save core file
shutil.copyfile('./eval.py', opt.model_dir + '/files/eval.py')
shutil.copyfile('./network_handdagt_tgdnet.py', opt.model_dir + '/network_handdagt_tgdnet.py')
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
							filename=os.path.join(opt.model_dir, 'train.log'), level=logging.INFO)
logging.info('======================================================')
# 综合方案（自动生成带时间戳的日志+数据集信息）
writer = SummaryWriter(
	f"runs/{opt.dataset}_" +
	f"{opt.model_name}_" +
	datetime.now().strftime("%Y%m%d_%H%M%S")
)



opt.result_file = open(opt.model_dir + '/result.txt', 'w')
opt.id_file = open(opt.model_dir + '/id.txt', 'w')
shutil.rmtree(opt.model_dir + '/img')
os.mkdir(opt.model_dir + '/img')

print('#Test data:', len(test_data))
print (opt)

# 2. Define model, loss
model = getattr(module, 'HandModel')(joints=opt.JOINT_NUM, stacks=opt.stacks, dataset=opt.dataset)

if opt.ngpu > 1:
	model.netR_1 = torch.nn.DataParallel(model.netR_1, range(opt.ngpu))
	model.netR_2 = torch.nn.DataParallel(model.netR_2, range(opt.ngpu))
	model.netR_3 = torch.nn.DataParallel(model.netR_3, range(opt.ngpu))
if opt.model != '':
	model_dir = opt.model_checkpoint_dir
	print('loading model from %s' % model_dir)
	print('model pth : %s' %opt.model)
	model.load_state_dict(torch.load(os.path.join(model_dir, opt.model)), strict=False)
		
model.cuda()
# print(model)

criterion = nn.MSELoss(size_average=True).cuda()


# 3. evaluation
torch.cuda.synchronize()

model.eval()
test_mse = 0.0
test_wld_err = 0.0
test_wld_err_mean = 0.0
#1表示没有残缺的关节点的误差，2表示残缺的关节点的误差
test1_wld_err = 0.0
test1_wld_err_sum = 0.0
test2_wld_err = 0.0
test2_wld_err_sum = 0.0

timer = 0

inc_total = 0

saved_points = []
saved_gt = []
saved_fold1 = []
saved_final = []
saved_error = []
saved_length = []

for i, data in enumerate(tqdm(test_dataloader, 0)):

	torch.cuda.synchronize()
	with torch.no_grad():
		# 3.2.1 load inputs and targets
		if opt.dataset == "nyu":
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length = data
			volume_length = volume_length.cuda()
		elif "IncNYU" in opt.dataset :
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para, volume_length, mask = data
			volume_length = volume_length.cuda()
			mask = mask.cuda()
		else:
			img, points, gt_xyz, uvd_gt, center, M, cube, cam_para = data
			volume_length = 250.

		points, gt_xyz, img = points.cuda(),  gt_xyz.cuda(), img.cuda()
		center, M, cube, cam_para = center.cuda(), M.cuda(), cube.cuda(), cam_para.cuda()

		# 风格迁移
		if ("IncNYU" in opt.dataset):
			hand_depth = img.float()
			transfer = transfer_img(".")
			transfered_hand_depth = transfer.transfer_syn(hand_depth)

			img = transfered_hand_depth

		t = time.time()
		estimation = model(points.transpose(1,2), points.transpose(1,2), img, test_data, center, M, cube, cam_para)
		timer += time.time() - t

	torch.cuda.synchronize()

	outputs_xyz = estimation.transpose(1, 2)

	#保存估计出的joint_xyz
	# print(outputs_xyz.shape)
	# print(center.shape)
	# print(cube.shape)
	outputs_xyz_np = outputs_xyz.detach().cpu().numpy()
	center_np = center.detach().cpu().numpy()
	cube_size_np = cube.detach().cpu().numpy()
	batchsize, joint_num, _ = outputs_xyz.shape
	center_np = np.tile(center_np.reshape(batchsize, 1, -1), [1, joint_num, 1])
	cube_size_np = np.tile(cube_size_np.reshape(batchsize, 1, -1), [1, joint_num, 1])
	joint_xyz_np = outputs_xyz_np * cube_size_np / 2 + center_np

	joint_xyz_save = joint_xyz_np[:, calculate, :]

	np.savetxt(opt.result_file, test_data.joint3DToImg(joint_xyz_save).reshape([batchsize, 14 * 3]), fmt='%.3f')


	joint_uvd = test_data.xyz_nl2uvdnl_tensor(outputs_xyz, center, M, cube, cam_para)
	# if (opt.result_file is not None):
	# 	np.savetxt(opt.result_file, test_data.jointImgTo3D(joint_uvd.cpu().numpy()).reshape([outputs_xyz.shape[0], opt.JOINT_NUM * 3]), fmt='%.3f')
	diff = torch.pow(outputs_xyz-gt_xyz, 2).view(-1,opt.JOINT_NUM, 3)
	diff_sum = torch.sum(diff, 2)
	diff_sum_sqrt = torch.sqrt(diff_sum)  #B, J
	if opt.JOINT_NUM == 23:
		diff_sum_sqrt = diff_sum_sqrt[:, calculate]
	diff_mean = torch.mean(diff_sum_sqrt,1).view(-1, 1)		# B, 1
	diff_mean_wld = torch.mul(diff_mean, volume_length / 2 if opt.dataset == 'dexycb' else volume_length.view(-1, 1) / 2)
	# print(diff_mean_wld)
	test_wld_err_mean = test_wld_err_mean + diff_mean_wld.sum().item()
	writer.add_scalar('error_mean', diff_mean_wld.mean().item(), global_step=i)
	# print(img[0].max())
	# print(img[0].min())
	# print(joint_uvd[0])
	
	# 对每一张图片都进行可视化并保存
	for batch_idx in range(img.size(0)):
		# 确定数据集类型用于可视化
		if joint_uvd.size(1) == 23:
			dataset_vis = 'nyu_all'
		else:
			dataset_vis = 'mano'
		
		# 生成预测结果的可视化
		img_show_pred = vis_tool.draw_2d_pose(img[batch_idx], joint_uvd[batch_idx], dataset_vis)
		
		# 生成真实标签的可视化
		img_show_gt = vis_tool.draw_2d_pose(img[batch_idx], uvd_gt[batch_idx], dataset_vis)
		
		# 计算当前样本的误差信息用于文件名
		current_error = diff_mean_wld[batch_idx].item()
		error_category = "high_error" if current_error > 9 else "normal"
		
		# 保存预测结果图片（包含更多信息）
		pred_img_path = os.path.join(opt.model_dir, 'img', 
			f'batch_{i:04d}_sample_{batch_idx:02d}_pred_{error_category}_error_{current_error:.2f}.png')
		print(pred_img_path)
		cv2.imwrite(pred_img_path, img_show_pred)
		
		# 保存真实标签图片
		gt_img_path = os.path.join(opt.model_dir, 'img', 
			f'batch_{i:04d}_sample_{batch_idx:02d}_gt_{error_category}_error_{current_error:.2f}.png')
		cv2.imwrite(gt_img_path, img_show_gt)
		
		# 同时保存到tensorboard（保持原有功能）
		if batch_idx == 0:  # 只将第一个样本添加到tensorboard
			writer.add_image('pd', np.transpose(img_show_pred, (2, 0, 1)) / 255.0, global_step=i)
			writer.add_image('gt', np.transpose(img_show_gt, (2, 0, 1)) / 255.0, global_step=i)
	if (diff_mean_wld > 9).sum() != 0:
		mask_tmp = diff_mean_wld.squeeze(1).cpu().numpy() > 9  # 添加维度压缩
		img_id = np.arange(img.size(0))[mask_tmp]
		img_id = opt.batchSize * i + img_id  # 修正参数名统一为batchSize
		np.savetxt(opt.id_file, img_id, fmt='%d')

	if 'IncNYU' in opt.dataset:
		outputs_xyz_mano = convert_nyu2mano(outputs_xyz)
		gt_xyz_mano = convert_nyu2mano(gt_xyz)
		diff = torch.pow(outputs_xyz_mano - gt_xyz_mano, 2).view(-1, 21, 3)
		# print(mask.shape)
		all_ones = (mask == 1).all(dim=-1, keepdim=True)
		joint_mask = all_ones.float()
		diff_1 = torch.mul(diff, mask)
		diff_2 = diff - diff_1
		diff_1_sum_sqrt = torch.sqrt(torch.sum(diff_1, 2))
		diff_2_sum_sqrt = torch.sqrt(torch.sum(diff_2, 2))
		count_1 = torch.sum(joint_mask, dim=1)
		# print(count_1)
		B = count_1.shape[0]
		diff_1_sum = torch.sum(diff_1_sum_sqrt, 1).view(-1, 1)  # 64x1
		diff_1_sum_wld = torch.mul(diff_1_sum,
								   volume_length / 2 if opt.dataset == 'dexycb' else volume_length.view(-1, 1) / 2)
		constant = torch.full_like(count_1, 21.0)
		count_2 = constant - count_1
		inc_total = inc_total + torch.sum(count_2, dim=0).item()
		B = count_2.shape[0]
		diff_2_sum = torch.sum(diff_2_sum_sqrt, 1).view(-1, 1)
		diff_2_sum_wld = torch.mul(diff_2_sum,
								   volume_length / 2 if opt.dataset == 'dexycb' else volume_length.view(-1, 1) / 2)
		test1_wld_err_sum = test1_wld_err_sum + diff_1_sum_wld.sum().item()
		test2_wld_err_sum = test2_wld_err_sum + diff_2_sum_wld.sum().item()

# print(test_wld_err_mean * 23)
print(test1_wld_err_sum)
print(test2_wld_err_sum)
# time taken
torch.cuda.synchronize()
print(len(test_data))
# timer = time.time() - timer
timer = timer / len(test_data)

print('==> time to learn 1 sample = %f (ms)' %(timer*1000))
print(inc_total)
# print mse
print('average estimation error in world coordinate system: ')
print(test_wld_err / len(test_data))
print(test_wld_err_mean / len(test_data))
print(test1_wld_err / (len(test_data)*21 - inc_total) )
print(test1_wld_err_sum / (len(test_data)*21 - inc_total))
print(test2_wld_err / inc_total)
print(test2_wld_err_sum / inc_total)

