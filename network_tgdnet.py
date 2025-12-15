
import torch
import torch.nn as nn
import math
import numpy as np
from pointutil import Conv1d,  Conv2d, PointNetSetAbstraction, BiasConv1d, square_distance, index_points_group
import torch.nn.functional as F
from convNeXT.resnetUnet import convNeXTUnetBig
from transformer import Transformer, Mlp
# from pointnet2 import pointnet2_utils
from pointNet.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils

from util.vis_tool import get_hierarchy_sketch, get_hierarchy_mapping, get_sketch_setting
from util.graph_util import adj_mx_from_edges
from semGCN.sem_gcn import SimpleSemGCN
from semGCN.ph_gcn import SPGCN
from render.obman_mano import Render
import time

def smooth_l1_loss(input, target, sigma=10., reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.sum(loss, dim=1) / normalizer
criterion = smooth_l1_loss

model_list = {
          'tiny': ([3, 3, 9, 3], [96, 192, 384, 768]),
          'small': ([3, 3, 27, 3], [96, 192, 384, 768]),
          'base': ([3, 3, 27, 3], [128, 256, 512, 1024]),
          'large': ([3, 3, 27, 3], [192, 384, 768, 1536])
          }
weight_url_1k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224.pth"
}

weight_url_22k = {
    'tiny': "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    'small': "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    'base': "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    'large': "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth"
}

class CurvatureCalculator(nn.Module):
    def __init__(self, radius, nsample=6):
        super(CurvatureCalculator, self).__init__()
        self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=True)

    def forward(self, pcl_coord, joint_coord):
        B, J, N, _ = pcl_coord.shape

        # 重塑输入以匹配 QueryAndGroup 的预期输入
        pcl_coord_reshaped = pcl_coord.reshape(B * J, N, 3)
        joint_coord_reshaped = joint_coord.reshape(B * J, 1, 3)

        # 使用 QueryAndGroup 找到邻域点
        grouped_points = self.queryandgroup(pcl_coord_reshaped, joint_coord_reshaped)

        # grouped_points 的形状是 (B*J, 3, 1, nsample)
        # 我们需要重塑它以方便后续计算
        grouped_points = grouped_points.squeeze(2).transpose(1, 2)  # (B*J, nsample, 3)

        # 计算拉普拉斯-贝尔特拉米算子
        v1 = grouped_points[:, :, None, :] - grouped_points[:, None, :, :]
        v2 = v1.transpose(1, 2)

        dot_product = torch.sum(v1 * v2, dim=-1)
        v1_norm = torch.norm(v1, dim=-1)
        v2_norm = torch.norm(v2, dim=-1)
        cos_angle = dot_product / (v1_norm * v2_norm + 1e-6)
        sin_angle = torch.norm(torch.cross(v1, v2, dim=-1), dim=-1) / (v1_norm * v2_norm + 1e-6)
        cot_alpha = cos_angle / (sin_angle + 1e-6)

        laplacian = torch.sum(v1 * cot_alpha.unsqueeze(-1) / 2, dim=(1, 2))

        # 计算曲率方向向量
        # 曲率向量尚未归一化
        curvature_dir = laplacian

        # 计算曲率的大小（曲率向量的模长）
        curvature_magnitude = torch.norm(curvature_dir, dim=-1, keepdim=True)

        # 计算归一化的曲率方向向量
        curvature_dir_normalized = curvature_dir / (curvature_magnitude + 1e-6)

        # 重塑结果以匹配预期的输出形状
        curvature_dir_normalized = curvature_dir_normalized.view(B, J, 3)
        curvature_magnitude = curvature_magnitude.view(B, J)

        # 扩展 curvature_magnitude 的维度以与 curvature_dir_normalized 相乘
        curvature_magnitude = curvature_magnitude.unsqueeze(-1)  # (B, J, 1)

        # 两者相乘，得到最终的曲率向量
        curvature_vector = curvature_dir_normalized * curvature_magnitude  # (B, J, 3)

        return curvature_vector

class PoolSPGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, joint_num, dataset='nyu_all'):
        super().__init__()
        adjs = []
        for ajd in get_hierarchy_sketch(dataset):
            bone_num = np.array(ajd).max() + 1
            adjs.append(adj_mx_from_edges(bone_num, ajd, sparse=False, eye=True))
        node_maps = get_hierarchy_mapping(dataset)
        self.gcn = SPGCN(adjs, node_maps, in_dim, hid_dim, out_dim)

    def forward(self, x):
        x = self.gcn(x)
        return x

class TGDNet(nn.Module):
    def __init__(self, nsample, points2_channel, points1_channel, joint_num, mlp, mlp2=None, bn = False, use_leaky = True, radius=None, relu=False, bias=True, graph_bias=True):
        super(TGDNet,self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.mlp_q_convs = nn.ModuleList()
        self.mlp_g_convs = nn.ModuleList()
        self.mlp_v_convs = nn.ModuleList()
        self.mlp_k_convs = nn.ModuleList()
        self.mask_mlp = nn.ModuleList()
        self.mlp_q_bns = nn.ModuleList()
        self.mlp_g_bns = nn.ModuleList()
        self.mlp_v_bns = nn.ModuleList()
        self.mlp_k_bns = nn.ModuleList()
        self.mask_bns = nn.ModuleList()
        self.mlp2 = mlp2
        self.bn = bn
        self.relu = relu

        last_channel = points2_channel + 3

        #gcn相关模块
        self.graph_a = nn.Parameter(torch.randn(1, points1_channel, joint_num, joint_num).cuda(), requires_grad=True)
        self.graph_w = nn.Sequential(nn.Conv1d(points1_channel, points1_channel, 1, bias=graph_bias),
                                     nn.BatchNorm1d(points1_channel) if bn else nn.Identity(),
                                     nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True))
        self.fuse_q = nn.Conv1d(points1_channel, mlp[0], 1, bias=False)

        # #曲率特征编码
        # self.fuse_cur = nn.Conv1d(3, mlp[0], 1, bias=False)

        self.fuse_k = nn.Conv1d(points1_channel, mlp[0], 1, bias=False)
        self.fuse_v1 = nn.Conv2d(last_channel, mlp[0], 1, bias=False)
        self.fuse_v2 = nn.Conv2d(points1_channel, mlp[0], 1, bias=False)

        for i, out_channel in enumerate(mlp):
            self.mlp_q_convs.append(nn.Conv2d(last_channel, out_channel if i < len(mlp)-1 else out_channel * 2, 1, bias=bias))
            self.mlp_v_convs.append(nn.Conv2d(last_channel if i > 0 else mlp[0], out_channel, 1, bias=bias))
            self.mlp_k_convs.append(nn.Conv2d(last_channel, out_channel if i < len(mlp)-1 else out_channel * 2, 1, bias=bias))
            self.mask_mlp.append(nn.Conv1d(last_channel, out_channel if i < len(mlp)-1 else 1
                                           , 1, bias=False))
            if bn:
                self.mlp_q_bns.append(nn.BatchNorm2d(out_channel if i < len(mlp)-1 else out_channel * 2))
                self.mlp_v_bns.append(nn.BatchNorm2d(out_channel))
                self.mlp_k_bns.append(nn.BatchNorm2d(out_channel if i < len(mlp)-1 else out_channel * 2))
                self.mask_bns.append(nn.BatchNorm2d(out_channel))

            last_channel = out_channel

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(0.1, inplace=True)

        #聚合已有关节点的传统transformer
        self.transform = Transformer(dim=512, depth=1, num_heads=4, mlp_ratio=2.)
        model = self.transform
        # print("层级参数明细：")
        # for name, param in model.named_parameters():
        #     print(f"{name:40} | {param.numel():,}")

        print(f"\n总参数量：{sum(p.numel() for p in model.parameters()):,}")
        # self.final_mlp = Mlp(in_features=512, hidden_features=1024, act_layer=nn.GELU)

        # #clossness_ratio
        # self.alpha = nn.Parameter(torch.tensor(0.3))

        # #计算曲率
        # self.curvature_guide = CurvatureCalculator(0.5, nsample=32);
        #
        # # 父亲节点对子节点的影响力 1表示缺失，0表示不缺失
        # # self.mask2mask_ratio = 0.3
        # self.mask2mask_ratio = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        # # self.mask2mask_ratio = nn.Parameter(torch.empty(1).uniform_(0, 1))  # 初始化范围 [0, 1]
        #
        # # mask对关节点估计的影响,残缺概率越大，g2就越大，g1就越小
        # # self.mask2j_ratio = 0.5
        # self.mask2j_ratio = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        # self.mask2j_ratio = nn.Parameter(torch.empty(1).uniform_(0, 1))  # 初始化范围 [0, 1]

        if radius is not None:
            self.queryandgroup = pointnet2_utils.QueryAndGroup(radius, nsample, True)

    def forward(self, xyz1, xyz2, points1, points2, xyz3=None, points3=None, mask3=None):
        '''
        add fuse_v
        xyz1: joints [B, 3, N1]
        xyz2: local points [B, 3, N2]
        points1: joints features [B, C, N1]
        points2: local features [B, C, N2]

        xyz3 : 已经估计好的关节点
        points3: 已经估计好的关节点的特征
        mask3 : 已经估计好的关节点的mask
        '''
        B, C, N1 = xyz1.shape   #32, 3, N1
        _, _, S1, _ = xyz2.shape   #64
        _, D1, _ = points1.shape    #512
        _, D2, _, _ = points2.shape    #256+3
        # print(D1)
        # print(S1)
        # print(points2.shape)
        # print(points1.shape)


        # xyz2 = xyz2.permute(0, 3, 2, 1)     # B, N1，S， 3
        # xyz1 = xyz1.permute(0, 2, 1)    # B, N1, 3

        # #计算关节点的曲率特征
        # # start_time = time.time()
        # curvature = self.curvature_guide(xyz2, xyz1)  # B, N1, 3
        # # print('curvature Time:%.3f' % ((time.time() - start_time) * 1000))
        # points1 = points1 + self.fuse_cur(curvature.permute(0, 2, 1))

        # #根据关节点到点云点的距离，计算关节点与点云点的关系
        # xyz1_expanded = xyz1.unsqueeze(2).expand(B, N1, S1, 3)
        # distance = torch.sum(torch.pow(xyz2 - xyz1_expanded, 2), dim=-1)  # B N S
        # closeness_value = 1 / (distance + 1e-8)
        # closeness_value_normal = closeness_value / (closeness_value.sum(-1, keepdim=True) + 1e-8)   # B, N, S
        # joint_dis = torch.sum(points1.permute(0, 1, 3, 2) * closeness_value_normal.unsqueeze(1), dim=-1)  # B C N

        # points2_max = torch.max(points2, -2)[0]
        # # print(points2_max.shape)
        # mask = points2_max
        # for i, conv in enumerate(self.mask_mlp):
        #     mask = conv(mask)
        #     if self.bn:
        #         mask = self.mask_bns[i](mask)
        #     if i == len(self.mlp_q_convs) - 1:
        #         mask = mask
        #     else:
        #         mask = self.relu(mask)
        # mask_now = self.sigmoid(mask)
        # # #增加父亲节点对子节点的影响
        # if mask3 is not None:
        #     # print(mask3.shape)
        #     # print(mask_now.shape)
        #     mask_now = mask_now + mask3*self.mask2mask_ratio
        #     # 限制范围在 (0, 1)
        #     mask_now = torch.clamp(mask_now, 0.0, 1.0)

        #gcn
        point1_graph = self.graph_w(torch.matmul(points1.unsqueeze(-2), self.graph_a).squeeze(-2))

        # # 根据attention机制计算已估计好的关节点与关节点的关系，矩阵乘法的attention
        # if xyz3 is not None:
        #     q = point1_graph.permute(0, 2, 1)
        #     k = points3.permute(0, 2, 1)
        #     # print(q.shape)
        #     # print(k.shape)
        #     v_pre = self.transform(q, k).permute(0, 2, 1)
        #     point1_graph = v_pre

        #根据attention机制计算关节点与点云点的关系,元素乘法的attention，计算关节点到点云点的关系
        # q:点云特征
        q = points2
        for i, conv in enumerate(self.mlp_q_convs):
            q = conv(q)
            if i == 0:
                grouped_points1 = self.fuse_q(point1_graph)
                q = q + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                q = self.mlp_q_bns[i](q)
            if i == len(self.mlp_q_convs) - 1:
                q = q
            else:
                q = self.relu(q)

        #k:关节点特征
        # points1 = self.fuse_k(points1)
        # k = points1.view(B, points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        k = points2
        for i, conv in enumerate(self.mlp_k_convs):
            k = conv(k)
            if i == 0:
                grouped_points1 = self.fuse_k(point1_graph)
                k = k + grouped_points1.view(B, grouped_points1.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
            if self.bn:
                k = self.mlp_k_bns[i](k)
            if i == len(self.mlp_k_convs) - 1:
                k = k
            else:
                k = self.relu(k)
            if i == len(self.mlp_k_convs) - 2:
                k = torch.max(k, -2)[0].unsqueeze(-2)

        #点云点与关节点的关系，距离+attention自适应
        a = self.sigmoid(k * q)     #B, C, S, N1
        g1, g2 = torch.chunk(a, 2, 1)
        # b = closeness_value_normal.permute(0, 2, 1).unsqueeze(1)    #B, 1, S, N1
        # alpha = torch.sigmoid(self.alpha)  # 初始化一个可学习参数 self.alpha
        # c = alpha * a + (1 - alpha) * b
        # # c = (a + b) / 2           #B, C, S, N1， 除以2缩放回[0, 1]
        # g1, g2 = torch.chunk(c, 2, 1)


        # print(g1.shape)
        # print(mask_now.shape)
        # mask_now_expanded = mask_now.unsqueeze(2) # 扩展到 (B, 1, 1, N1)
        # mask_now_expanded = mask_now_expanded.expand(B, D1, S1, N1)  # 扩展到 (B, C, S, N1)
        # g1 = g1 - mask_now_expanded*self.mask2j_ratio
        # g2 = g2 + mask_now_expanded*self.mask2j_ratio

        #v：根据点云点和关节点的关系得到新的关节点特征
        v = points2
        v = self.fuse_v1(v)

        point1_graph_expand = point1_graph.view(B, point1_graph.size(1), 1, N1).repeat(1, 1, self.nsample, 1)
        point1_expand = g2 * point1_graph_expand
        point1_expand = self.fuse_v2(point1_expand)

        v = self.relu(v * g1 + point1_expand) + points1.unsqueeze(2).repeat(1, 1, self.nsample, 1)
        v_res = v.mean(2)

        #前向传播
        for i, conv in enumerate(self.mlp_v_convs):
            v = conv(v)
            if i == 0:
                v = v * g1 + point1_expand
            if self.bn:
                v = self.mlp_v_bns[i](v)
            if i == len(self.mlp_v_convs) - 1:
                v = self.relu(v)
            else:
                v = self.relu(v)
            if i == len(self.mlp_v_convs) - 2:
                v = torch.max(v, -2)[0].unsqueeze(-2)

        v = v.squeeze(-2)
        v = v + v_res


        # # 根据附近点云点和关节点之间的关系，估计关节点的残缺概率
        # c_mask = torch.max(c, -2)[0]  # B, C, N1
        # c_mask = self.mask_mlp(c_mask.permute(0, 2, 1)).permute(0, 2, 1)


        # #根据残缺情况判断融合已有关节点的比例
        # ratio = c_mask.squeeze(-2)
        # for i, fc in enumerate(self.fuse_mask):
        #     ratio = fc(ratio)
        #     if i == len(self.mlp_mask) - 1:
        #         ratio = ratio
        #     else:
        #         ratio = self.relu(ratio)
        #
        # ratio = self.sigmoid(ratio)

        # c_mask = c_mask.squeeze(-2)   #B, 1, N1

        # # 根据attention机制计算已估计好的关节点与关节点的关系，矩阵乘法的attention
        # if xyz3 is not None:
        #     q = v.permute(0, 2, 1)
        #     k = points3.permute(0, 2, 1)
        #     v_pre = self.transform(q, k).permute(0, 2, 1)
        #     # v = v + v_pre
        #     v = v_pre
        #     # v_for = self.final_mlp(v.permute(0, 2, 1)).permute(0, 2, 1)
        #     # v = v + v_for

        # print(v.shape)
        # print(c_mask.shape)
        # print(mask_now.shape)
        # return v, mask_now
        return v

class HandModel(nn.Module):
    def __init__(self, joints=21, stacks=10, dataset='nyu', share_regress=False, teacher_model_path=None):
        super(HandModel, self).__init__()
        dataset = 'nyu'
        if 'IncNYU' in dataset:
            self.levels_num = 4
            self.levels = [[0, 13, 1, 4, 10, 7], [14, 2, 5, 11, 8], [15, 3, 6, 12, 9], [20, 16, 17, 19, 18]]
            self.Diff2GT = [0, 2, 7, 12, 3, 8, 13, 5, 10, 15, 4, 9, 14, 1, 6, 11, 17, 18, 20, 19, 16]
            self.joints_num = 21
        elif dataset == 'nyu_all':
            self.levels_num = 4
            self.levels = [[21, 22, 20, 3, 7, 11, 15, 19], [2, 6, 10, 14, 18], [1, 5, 9, 13, 17], [0, 4, 8, 12, 16]]
            #18, 13, 8, 3
            self.Diff2GT = [18, 13, 8, 3, 19, 14, 9, 4, 20, 15, 10, 5, 21, 16, 11, 6, 22, 17, 12, 7, 2, 0, 1]
            self.joints_num = 23
        elif dataset == 'nyu':
            self.levels_num = 4
            self.levels = [[21, 22, 20, 3, 7, 11, 15, 19], [2, 6, 10, 14, 18], [1, 5, 9, 13, 17], [0, 4, 8, 12, 16]]
            # 18, 13, 8, 3
            self.Diff2GT = [18, 13, 8, 3, 19, 14, 9, 4, 20, 15, 10, 5, 21, 16, 11, 6, 22, 17, 12, 7, 2, 0, 1]
            self.joints_num = 23

        self.backbone = convNeXTUnetBig('small', pretrain='1k', deconv_dim=128)


        #给图像点进行采样
        self.encoder_0 = PointNetSetAbstraction(npoint=256, radius=0.1, nsample=64, in_channel=128, mlp=[128,128,128])

        self.encoder_1 = PointNetSetAbstraction(npoint=256, radius=0.1, nsample=64, in_channel=3, mlp=[32,32,128])

        self.encoder_2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=64, in_channel=128, mlp=[128,64,256])

        self.encoder_3 = nn.Sequential(Conv1d(in_channels=256+3, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=128, bn=True, bias=False),
                                       Conv1d(in_channels=128, out_channels=512, bn=True, bias=False),
                                       nn.MaxPool1d(128,stride=1))

        self.fold1 = nn.Sequential(BiasConv1d(bias_length=joints, in_channels=512+768, out_channels=512, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True),
                                    BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True))
        # #估计mask
        # self.fold2 = nn.Sequential(BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True),
        #                            BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True),
        #                            BiasConv1d(bias_length=joints, in_channels=512, out_channels=1, bn=True))
        # self.loss_mask = torch.nn.BCELoss().cuda()
        self.sigmoid = nn.Sigmoid()

        # self.mask_ratio = 100

        self.regress_1 = nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)

        self.trans = nn.ModuleList()
        self.regress = nn.ModuleList() if not share_regress else None
        for i in range(stacks):
            trans_stack = nn.ModuleList([
                TGDNet(
                    nsample=64,
                    points2_channel=256,
                    points1_channel=512,
                    joint_num=len(self.levels[i]),
                    mlp=[512, 512, 512],
                    mlp2=None
                ) for i in range(self.levels_num)
            ])
            self.trans.append(trans_stack)
        # for i in range(stacks):
        #     trans_stack = TGDNet(
        #             nsample=64,
        #             points2_channel=256,
        #             points1_channel=512,
        #             joint_num=len(self.levels[i]),
        #             mlp=[512, 512, 512],
        #             mlp2=None
        #         )
        #     self.trans.append(trans_stack)
            if not share_regress:
                regress_stack = nn.ModuleList([
                    nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)
                    for _ in range(self.levels_num)
                ])
                # regress_stack = nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)

                self.regress.append(regress_stack)

        if share_regress:
            self.regress = nn.ModuleList([
                nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)
                for _ in range(self.levels_num)
            ])

            # self.regress = nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)


        self.share_regress = share_regress
        self.stacks = stacks
        self.joints = joints

        # # mano模型回归
        # mano_dir = './MANO/'
        # self.render = Render(mano_dir, dataset)
        #
        # self.PoolGCN = PoolSPGCN(512, 512, 512, self.joints_num, dataset="mano")
        # self.mano_reg = nn.Sequential(nn.Linear(512, 3 + 45 + 10 + 4))
        # self.mesh_ratio = 1e-3
        # self.pose_ratio = 1
        # self.shape_ratio = 1
        # self.mano_joint_ratio = 1e-1

        if teacher_model_path is not None:
            # teacher_encoder
            # 加载权重到模型中
            encoder_related_state_dict = torch.load(teacher_model_path)

            self.teacher_backbone = convNeXTUnetBig('small', pretrain='1k', deconv_dim=128)
            self.teacher_encoder_1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=64, in_channel=3,
                                                            mlp=[32, 32, 128])

            self.teacher_encoder_2 = PointNetSetAbstraction(npoint=128, radius=0.3, nsample=64, in_channel=128,
                                                            mlp=[64, 64, 256])

            self.teacher_encoder_3 = nn.Sequential(Conv1d(in_channels=256 + 3, out_channels=128, bn=True, bias=False),
                                                   Conv1d(in_channels=128, out_channels=128, bn=True, bias=False),
                                                   Conv1d(in_channels=128, out_channels=512, bn=True, bias=False),
                                                   nn.MaxPool1d(128, stride=1))

            self.teacher_fold1 = nn.Sequential(
                BiasConv1d(bias_length=joints, in_channels=512 + 768, out_channels=512, bn=True),
                BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True),
                BiasConv1d(bias_length=joints, in_channels=512, out_channels=512, bn=True))
            self.teacher_regress_1 = nn.Conv1d(in_channels=512, out_channels=3, kernel_size=1)

            self.teacher_encoder_1.load_state_dict(
                {k.replace("encoder_1.", ""): v for k, v in encoder_related_state_dict.items() if
                 k.startswith("encoder_1.")})
            self.teacher_encoder_2.load_state_dict(
                {k.replace("encoder_2.", ""): v for k, v in encoder_related_state_dict.items() if
                 k.startswith("encoder_2.")})
            self.teacher_encoder_3.load_state_dict(
                {k.replace("encoder_3.", ""): v for k, v in encoder_related_state_dict.items() if
                 k.startswith("encoder_3.")})
            # 处理教师模型的权重字典，去除backbone前缀
            new_backbone_state_dict = {}
            for key in encoder_related_state_dict.keys():
                if key.startswith('backbone.'):
                    new_key = key[len('backbone.'):]  # 去除前缀
                    new_backbone_state_dict[new_key] = encoder_related_state_dict[key]
            self.teacher_backbone.load_state_dict(new_backbone_state_dict)

            self.teacher_fold1.load_state_dict(
                {k.replace("fold1.", ""): v for k, v in encoder_related_state_dict.items() if k.startswith("fold1.")})
            self.teacher_regress_1.load_state_dict(
                {k.replace("regress_1.", ""): v for k, v in encoder_related_state_dict.items() if
                 k.startswith("regress_1.")})
            for module in [self.teacher_encoder_1, self.teacher_encoder_2, self.teacher_encoder_3, self.teacher_backbone,
                           self.teacher_fold1, self.teacher_regress_1]:
                for param in module.parameters():
                    param.requires_grad = False
            print("Encoder-related parameters have been frozen.")
            self.teacher_loss_ratio = 1e-5
            
            # 关系蒸馏损失权重配置 (使用Smooth L1 Loss)
            self.pairwise_relation_ratio = 1e-2    # 直接关节对关系损失权重 (Smooth L1需要更大权重)
            self.cosine_relation_ratio = 0     # 余弦相似度关系损失权重
            self.euclidean_relation_ratio = 0   # 欧氏距离关系损失权重
            
            print(f"Teacher loss ratio: {self.teacher_loss_ratio}")
            print(f"Relation loss ratios - Pairwise: {self.pairwise_relation_ratio}, Cosine: {self.cosine_relation_ratio}, Euclidean: {self.euclidean_relation_ratio}")

    def log_snr(self, t):
        return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))
    def log_snr_to_alpha_sigma(self, log_snr):
        return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

    def encode(self, pc, feat, img, loader, center, M, cube, cam_para):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1

        pc1, feat1 = self.encoder_1(pc, feat)  # B, 3, 512; B, 64, 512

        pc2, feat2 = self.encoder_2(pc1, feat1)  # B, 3, 128; B, 256, 128

        code = self.encoder_3(torch.cat((pc2, feat2), 1))  # B, 512, 1 点云的全局特征

        pc_img_feat, c4 = self.backbone(img)  # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
        img_code = torch.max(c4.view(c4.size(0), c4.size(1), -1), -1, keepdims=True)[0]
        B, C, H, W = pc_img_feat.size()
        img_down = F.interpolate(img, [H, W])
        B, _, N = pc1.size()

        pcl_closeness, pcl_index, img_xyz = loader.img2pcl_index(pc1.transpose(1, 2).contiguous(), img_down, center, M,
                                                                 cube, cam_para, select_num=4)

        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)  # B*128*(K*1024)
        pcl_feat = torch.gather(pc_img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat = torch.sum(pcl_feat * pcl_closeness.unsqueeze(1), dim=-1)
        feat1 = torch.cat((feat1, pcl_feat), 1)

        code = code.expand(code.size(0), code.size(1), self.joints)
        img_code = img_code.expand(img_code.size(0), img_code.size(1), self.joints)

        latents = self.fold1(torch.cat((code, img_code), 1))
        joints = self.regress_1(latents)

        return latents, joints, pc1, feat1, img_xyz, pc_img_feat

    # def encode(self, pc, feat, img, loader, center, M, cube, cam_para):
    #     # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
    #
    #     pc1, feat1 = self.encoder_1(pc, feat)# B, 3, 512; B, 128, 512
    #
    #     pc2, feat2 = self.encoder_2(pc1, feat1)# B, 3, 128; B, 256, 128
    #
    #     code = self.encoder_3(torch.cat((pc2, feat2),1))# B, 512, 1 点云的全局特征
    #
    #     pc_img_feat, c4 = self.backbone(img)   # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
    #     img_code = torch.max(c4.view(c4.size(0),c4.size(1),-1),-1,keepdims=True)[0] #128, 768, 1 图像全局特征
    #     B, C, H, W = pc_img_feat.size()  #128, 128, 64, 64
    #     img_down = F.interpolate(img, [H, W])
    #     B, _, N = pc1.size()
    #
    #     pcl_closeness, pcl_index, img_xyz = loader.img2pcl_index(pc1.transpose(1,2).contiguous(), img_down, center, M, cube, cam_para, select_num=4)
    #
    #     # pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)   # B*128*(K*1024)
    #     # pcl_feat = torch.gather(pc_img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
    #     # pcl_feat = torch.sum(pcl_feat*pcl_closeness.unsqueeze(1), dim=-1)
    #     pc_img_feat = pc_img_feat.view(B, C, -1)    #128, 128, 4096
    #     img_xyz = img_xyz.transpose(1, 2)   #128, 3, 4096
    #     # print(img_xyz.shape)
    #     # print(pc_img_feat.shape)
    #     img_xyz, img_feat = self.encoder_0(img_xyz, pc_img_feat)    #128, 3, 512; 128, 128, 512
    #
    #     feat1 = torch.cat((feat1, img_feat), -1)    #128, 128, 1024
    #     pc1 = torch.cat((pc1, img_xyz), -1)     #128, 3, 1024
    #     # print(feat1.shape)
    #     # print(pc1.shape)
    #
    #     code = code.expand(code.size(0),code.size(1), self.joints)
    #     img_code = img_code.expand(img_code.size(0),img_code.size(1), self.joints)
    #
    #     latents = self.fold1(torch.cat((code, img_code),1)) #B*512*21，joint-wise的全局特征，卷积的时候加入bias
    #
    #     # #初步估计mask
    #     # masks = self.fold2(latents)
    #     # masks = self.sigmoid(masks)
    #
    #     joints = self.regress_1(latents)
    #
    #     return latents, joints, pc1, feat1, img_xyz, pc_img_feat
    #     # return latents, joints, masks, pc1, feat1, img_xyz, pc_img_feat

    # teacher_encoder
    def teacher_encoder(self, pc, feat, img, loader, center, M, cube, cam_para):
        # x: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, y: B*3*sample_num_level1*1
        # #teacher_backbone
        # if teacher_img != None:
        #     teacher_pc_img_feat, teacher_c4 = self.teacher_backbone(teacher_img)

        pc1, feat1 = self.teacher_encoder_1(pc, feat)  # B, 3, 512; B, 64, 512

        pc2, feat2 = self.teacher_encoder_2(pc1, feat1)  # B, 3, 128; B, 256, 128

        code = self.teacher_encoder_3(torch.cat((pc2, feat2), 1))  # B, 512, 1 点云的全局特征

        pc_img_feat, c4 = self.teacher_backbone(
            img)  # img_offset: B×C×W×H , C=3(direct vector)+1(heatmap)+1(weight)
        img_code = torch.max(c4.view(c4.size(0), c4.size(1), -1), -1, keepdims=True)[0]
        B, C, H, W = pc_img_feat.size()
        img_down = F.interpolate(img, [H, W])
        B, _, N = pc1.size()

        pcl_closeness, pcl_index, img_xyz = loader.img2pcl_index(pc1.transpose(1, 2).contiguous(), img_down, center,
                                                                 M,
                                                                 cube, cam_para, select_num=4)

        pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)  # B*128*(K*1024)
        pcl_feat = torch.gather(pc_img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
        pcl_feat = torch.sum(pcl_feat * pcl_closeness.unsqueeze(1), dim=-1)
        feat1 = torch.cat((feat1, pcl_feat), 1)

        code = code.expand(code.size(0), code.size(1), self.joints)
        img_code = img_code.expand(img_code.size(0), img_code.size(1), self.joints)

        latents = self.teacher_fold1(torch.cat((code, img_code), 1))
        joints = self.teacher_regress_1(latents)

        return pc_img_feat, latents

    def forward(self, pc, feat, img, loader, center, M, cube, cam_para):
        embed, joint, pc1, feat1, _, _ = self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        # embed, joint, masks, pc1, feat1, _, _ = self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        mano_feat = None
        for i in range(self.stacks):
            points2_sample_feat, points2_sample_xyz = self.sample_points(joint, pc1, embed, feat1)    # [B, D2+3, nsample, N1], [B, D2+3, nsample, N1]
            points1_now = []
            xyz1_now = []
            # mask_now = []
            for j in range(self.levels_num):
                if j == 0:
                    points1_level_j = self.trans[i][j](joint[:, :, self.levels[j]],
                                                                       points2_sample_xyz[:, :, :, self.levels[j]],
                                                                       embed[:, :, self.levels[j]],
                                                                       points2_sample_feat[:, :, :, self.levels[j]])
                    # points1_level_j, points1_mask_j = self.trans[i][j](joint[:, :, self.levels[j]], points2_sample_xyz[:, :, :, self.levels[j]], embed[:, :, self.levels[j]], points2_sample_feat[:, :, :, self.levels[j]])
                else:
                    #得到已估计好的关节点和特征
                    xyz3 = torch.cat(xyz1_now, dim=-1)
                    points3 = torch.cat(points1_now, dim=-1)
                    # mask3 = torch.cat(mask_now, dim=-1)
                    # xyz3 = xyz3[:, :, -5:]
                    # points3 = points3[:, :, -5:]
                    # mask3 = mask3[:, :, -5:]
                    points1_level_j = self.trans[i][j](joint[:, :, self.levels[j]],
                                                                       points2_sample_xyz[:, :, :, self.levels[j]],
                                                                       embed[:, :, self.levels[j]],
                                                                       points2_sample_feat[:, :, :, self.levels[j]]
                                                                       , xyz3, points3)
                    # points1_level_j, points1_mask_j = self.trans[i][j](joint[:, :, self.levels[j]], points2_sample_xyz[:, :, :, self.levels[j]], embed[:, :, self.levels[j]], points2_sample_feat[:, :, :, self.levels[j]]
                    #                                     , xyz3, points3, mask3)
                if self.share_regress:
                    joint_j = self.regress[j](points1_level_j)
                else:
                    joint_j = self.regress[i][j](points1_level_j)
                #记录每层估计出的特征
                # mask_now.append(points1_mask_j)
                points1_now.append(points1_level_j)
                xyz1_now.append(joint_j)
                # # 得到已估计好的关节点和特征
                # points3 = torch.cat(points1_now, dim=-1)
                # mask3 = torch.cat(mask_now, dim=-1)
                # # xyz3 = xyz3[:, :, -5:]
                # # points3 = points3[:, :, -5:]
                # mask3 = mask3[:, :, -5:]
                # if self.share_regress:
                #     joint_j = self.regress[j](points3)
                # else:
                #     joint_j = self.regress[i][j](points3)
                # xyz3 = joint_j
            embed = torch.cat(points1_now, dim=-1)
            joint = torch.cat(xyz1_now, dim=-1)
            # embed = points3
            # joint = xyz3
            # mask = torch.cat(mask_now, dim=-1)
            embed = embed[:, :, self.Diff2GT]
            joint = joint[:, :, self.Diff2GT]
            # mask = mask[:, :, self.Diff2GT]

            # # 加入MANO
            # joint_feat = embed.transpose(1, 2)  # (B, J, C）
            # joint_xyz = joint.transpose(1, 2)  # (B, J, 3)
            # if mano_feat is not None:
            #     mano_feat = self.PoolGCN(joint_feat) + mano_feat
            # else:
            #     mano_feat = self.PoolGCN(joint_feat)
            # mano_para = self.mano_reg(mano_feat)
            # # Add hand mesh
            # render = self.render
            # mano_mesh, mano_joint = render.get_mesh(mano_para)
            # mano_joint = mano_joint
            # # print(mano_joint.shape)
            # mesh_center = torch.mean(mano_joint, dim=1, keepdim=True)
            # joint_center = torch.mean(joint_xyz, dim=1, keepdim=True)
            # # print(mesh_center.shape)
            # # print(joint_center.shape)
            # mano_mesh = mano_mesh - mesh_center + joint_center.detach()
            # mano_joint = mano_joint - mesh_center + joint_center.detach()
            #
            # # 更新joint
            # joint = mano_joint.transpose(1, 2)  # (B, 3, J)

        return joint
        # return joint, mask

    def get_loss(self, pc, feat, img, loader, center, M, cube, cam_para, gt_xyz, teacher_pcl, teacher_img, teacher_center,
                 teacher_M, teacher_cube):
        embed, joint, pc1, feat1, _, pc_img_feat = self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        # embed, joint, masks, pc1, feat1, _, _ = self.encode(pc, feat, img, loader, center, M, cube, cam_para)
        loss = smooth_l1_loss(joint, gt_xyz)
        # 特征蒸馏
        teacher_pc_image_feat, teacher_embed = self.teacher_encoder(teacher_pcl, teacher_pcl,
                                                                    teacher_img, loader,
                                                                    teacher_center, teacher_M,
                                                                    teacher_cube, cam_para)

        loss_embed = smooth_l1_loss(teacher_embed, embed) * self.teacher_loss_ratio * 1e3

        loss_pc_img_feat = smooth_l1_loss(teacher_pc_image_feat, pc_img_feat) * self.teacher_loss_ratio

        # 关节关系蒸馏损失设计
        B, C, J = embed.shape

        # 1. 直接关节对关系损失 - 显式建模关节对之间的特征差异
        teacher_embed_i = teacher_embed.unsqueeze(3).expand(B, C, J, J)  # [B, C, J, J]
        teacher_embed_j = teacher_embed.unsqueeze(2).expand(B, C, J, J)  # [B, C, J, J]
        teacher_pairwise = teacher_embed_i - teacher_embed_j  # [B, C, J, J] 教师网络关节对关系

        student_embed_i = embed.unsqueeze(3).expand(B, C, J, J)  # [B, C, J, J]
        student_embed_j = embed.unsqueeze(2).expand(B, C, J, J)  # [B, C, J, J]
        student_pairwise = student_embed_i - student_embed_j  # [B, C, J, J] 学生网络关节对关系

        # 使用Smooth L1损失对齐关节对关系 - 提供更好的鲁棒性
        pairwise_relation_loss = smooth_l1_loss(student_pairwise, teacher_pairwise) * self.pairwise_relation_ratio

        # 2. 余弦相似度关系 - 捕捉特征方向相似性
        teacher_embed_norm = F.normalize(teacher_embed, p=2, dim=1)
        student_embed_norm = F.normalize(embed, p=2, dim=1)

        teacher_cosine = torch.bmm(teacher_embed_norm.transpose(1, 2), teacher_embed_norm)
        student_cosine = torch.bmm(student_embed_norm.transpose(1, 2), student_embed_norm)
        cosine_loss = smooth_l1_loss(student_cosine, teacher_cosine) * self.cosine_relation_ratio

        # 3. 欧氏距离关系 - 捕捉特征空间距离
        teacher_euclidean = torch.norm(teacher_embed_i - teacher_embed_j, dim=1)  # [B, J, J]
        student_euclidean = torch.norm(student_embed_i - student_embed_j, dim=1)  # [B, J, J]
        euclidean_loss = smooth_l1_loss(student_euclidean, teacher_euclidean) * self.euclidean_relation_ratio

        # 聚合所有关系损失
        total_relation_loss = pairwise_relation_loss + cosine_loss + euclidean_loss

        # loss_embed = 0
        # loss_pc_img_feat = 0

        loss += loss_embed + loss_pc_img_feat + total_relation_loss

        # mano_feat = None
        # print("loss_joint: %f", loss)
        # masks_0 = masks.squeeze(1)
        # gt_mask_0 = gt_mask.squeeze(1)
        # print(masks_0)
        # print(gt_mask_0.dtype)
        # print("masks_0 shape:", masks_0.shape)
        # print("gt_mask_0 shape:", gt_mask_0.shape)
        # loss_mask = self.loss_mask(masks_0, gt_mask_0) * self.mask_ratio
        # print("loss_mask: %f", loss_mask)
        # loss += loss_mask

        times = torch.zeros(
            (joint.size(0),), device=joint.device).float().uniform_(0.5, 1)
        log_snr = self.log_snr(times)
        alpha, sigma = self.log_snr_to_alpha_sigma(times)
        # c0 = alpha.view(-1, 1, 1)   # (B, 1, 1)
        c1 = torch.sqrt(torch.sigmoid(-log_snr)).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(joint)  # (B, d, J)
        joint = joint + c1 * e_rand

        for i in range(self.stacks):
            points2_sample_feat, points2_sample_xyz = self.sample_points(joint, pc1, embed, feat1, nsample=64)    # [B, D2+3, nsample, N1], [B, D2+3, nsample, N1]
            points1_now = []
            xyz1_now = []
            # level_j = []
            # mask_now = []
            # print(self.trans[i])
            # print(points2_sample_xyz.shape)
            # print([points2_sample_feat.shape])
            for j in range(self.levels_num):
                if j == 0:
                    points1_level_j = self.trans[i][j](joint[:, :, self.levels[j]],
                                                                       points2_sample_xyz[:, :, :, self.levels[j]],
                                                                       embed[:, :, self.levels[j]],
                                                                       points2_sample_feat[:, :, :, self.levels[j]])
                    # points1_level_j, points1_mask_j = self.trans[i][j](joint[:, :, self.levels[j]], points2_sample_xyz[:, :, :, self.levels[j]], embed[:, :, self.levels[j]], points2_sample_feat[:, :, :, self.levels[j]])
                else:
                    #得到已估计好的关节点和特征
                    xyz3 = torch.cat(xyz1_now, dim=-1)
                    points3 = torch.cat(points1_now, dim=-1)
                    # mask3 = torch.cat(mask_now, dim=-1)
                    # xyz3 = xyz3[:, :, -5:]
                    # points3 = points3[:, :, -5:]
                    # mask3 = mask3[:, :, -5:]
                    # print(mask3.shape)
                    points1_level_j = self.trans[i][j](joint[:, :, self.levels[j]],
                                                       points2_sample_xyz[:, :, :, self.levels[j]],
                                                       embed[:, :, self.levels[j]],
                                                       points2_sample_feat[:, :, :, self.levels[j]]
                                                       , xyz3, points3)
                    # points1_level_j, points1_mask_j = self.trans[i][j](joint[:, :, self.levels[j]], points2_sample_xyz[:, :, :, self.levels[j]], embed[:, :, self.levels[j]], points2_sample_feat[:, :, :, self.levels[j]]
                    #                                     , xyz3, points3, mask3)

                if self.share_regress:
                    joint_j = self.regress[j](points1_level_j)
                else:
                    # print(points1_level_j.shape)
                    joint_j = self.regress[i][j](points1_level_j)
                # print(points1_level_j.shape)
                # print(joint_j.shape)
                # print(gt.shape)
                #计算joint_loss
                loss_joint_level = smooth_l1_loss(joint_j, gt_xyz[:, :, self.levels[j]])
                # print(loss_joint_level)
                loss += loss_joint_level
                #计算mask_loss
                # labels = gt_mask[:, :, self.levels[j]].squeeze(1)
                # points1_mask_j_s = points1_mask_j.squeeze(1)
                # weights = torch.where(labels == 0, 10.0, 1.0)  # 根据标签生成权重张量
                # criterion = nn.BCEWithLogitsLoss(reduction='none')
                # loss_bce = criterion(points1_mask_j, labels)  # 计算未加权的损失
                # weighted_loss = loss_bce * weights  # 加权
                # loss_mask = weighted_loss.mean()  # 求均值得到最终损失
                # loss_mask = self.loss_mask(points1_mask_j_s, labels) * (self.levels_num - i) * self.mask_ratio
                # loss += loss_mask
                #记录每层估计出的关节点和特征
                # mask_now.append(points1_mask_j)
                points1_now.append(points1_level_j)
                xyz1_now.append(joint_j)
                # # 得到已估计好的关节点和特征
                # points3 = torch.cat(points1_now, dim=-1)
                # mask3 = torch.cat(mask_now, dim=-1)
                # print(mask3.shape)
                # xyz3 = xyz3[:, :, -5:]
                # points3 = points3[:, :, -5:]
                # mask3 = mask3[:, :,  -5:]
                # if self.share_regress:
                #     joint_j = self.regress[j](points3)
                # else:
                #     # print(points1_level_j.shape)
                #     joint_j = self.regress[i][j](points3)
                # #计算joint_loss
                # level_j.extend(self.levels[j])
                # # print(level_j)
                # # print(gt_xyz.shape)
                # gt_xyz_j = gt_xyz[:, :, level_j]
                # # print(joint_j.shape)
                # # print(gt_xyz_j.shape)
                # loss += smooth_l1_loss(joint_j, gt_xyz_j)
                # xyz3 = joint_j
            embed = torch.cat(points1_now, dim=-1)
            joint = torch.cat(xyz1_now, dim=-1)
            # embed = points3
            # joint = xyz3
            embed = embed[:, :, self.Diff2GT]
            joint = joint[:, :, self.Diff2GT]

            # # 加入MANO
            # joint_feat = embed.transpose(1, 2)  #(B, J, C）
            # joint_xyz = joint.transpose(1, 2)   #(B, J, 3)
            # if mano_feat is not None:
            #     mano_feat = self.PoolGCN(joint_feat) + mano_feat
            # else:
            #     mano_feat = self.PoolGCN(joint_feat)
            # mano_para = self.mano_reg(mano_feat)
            # # Add hand mesh
            # render = self.render
            # mano_mesh, mano_joint = render.get_mesh(mano_para)
            # mano_joint = mano_joint
            # # print(mano_joint.shape)
            # mesh_center = torch.mean(mano_joint, dim=1, keepdim=True)
            # joint_center = torch.mean(joint_xyz, dim=1, keepdim=True)
            # # print(mesh_center.shape)
            # # print(joint_center.shape)
            # mano_mesh = mano_mesh - mesh_center + joint_center.detach()
            # mano_joint = mano_joint - mesh_center + joint_center.detach()
            #
            # #更新joint
            # joint = mano_joint.transpose(1, 2)  #(B, 3, J)
            #
            # mano_loss = smooth_l1_loss
            # loss_mano_joint = smooth_l1_loss(mano_joint, joint_xyz) * self.mano_joint_ratio
            # # print("loss_mano_joint {}".format(loss_mano_joint))
            # loss_mesh = mano_loss(mano_mesh, gt_mano_mesh) * self.mesh_ratio    #
            # # print("loss_mano_mesh {}".format(loss_mesh))
            # loss_pose = mano_loss(mano_para[:, 3:48], gt_mano_para[:, 3:48]) * self.pose_ratio  #
            # # print("loss_mano_pose {}".format(loss_pose))
            # loss_shape = mano_loss(mano_para[:, 48:58], gt_mano_para[:, 48:58]) * self.shape_ratio  #
            # # print("loss_mano_shape {}".format(loss_shape))
            # # loss_scale = torch.mean(torch.abs(torch.min(mano_para[:, 61:62], torch.zeros_like(mano_para[:, 61:62]).to(img.device))))
            # loss += loss_mano_joint + loss_pose + loss_shape + loss_mesh
        return loss

    def sample_points(self, xyz1, xyz2, points1, points2, nsample=64):
        '''
       add fuse_v
       xyz1: joints [B, 3, N1]
       xyz2: local points [B, 3, N2]
       points1: joints features [B, C, N1]
       points2: local features [B, C, N2]
       '''
        B, C, N1 = xyz1.shape  # 32, 3, 21
        _, _, N2 = xyz2.shape  # 512
        _, D1, _ = points1.shape  # 512
        _, D2, _ = points2.shape  # 256
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        sqrdists = square_distance(xyz1, xyz2)
        dists, knn_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2.permute(0, 2, 1), knn_idx)  # B, N1, nsample, D2
        new_points = torch.cat([grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D2+3, nsample, N1]
        neighbor_xyz = neighbor_xyz.permute(0, 3, 2, 1)  # [B, 3, nsample, N1]
        return new_points, neighbor_xyz