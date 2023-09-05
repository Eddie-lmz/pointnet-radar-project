'''
自己写的,用来测试
可视化文件夹下的点云数据
输入:n*3的矩阵
'''
import importlib
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import os

# with open('./point_test.pts') as file:
#     for line in file:
#         print(len(line))

import matplotlib.pylab as plt
import sys
# from utils.show_seg import seg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# Cyclinder=np.loadtxt('./point_test.pts')
points=np.loadtxt('data/modelnet40_normal_resampled/Cyclinder/Cyclinder_0001.txt',dtype=np.float32)  #预测只能输入float32的格式的数据
print(points.shape)

# 可视化
cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
# gt = cmap[seg.numpy() - 1, :]

# 可视化点云
showpoints(points)

#采样到2500个点
choice = np.random.choice(len(points), 2500, replace=True)
# print('choice:{}'.format(choice))
points = points[choice, :]
print('Cyclinder[choice, :]:{}'.format(points))
point_np=points



# 载入模型
# state_dict = torch.load('./seg/seg_model_Chair_1.pth')
# classifier = PointNetDenseCls(k= state_dict['conv4.weight'].size()[0])
# classifier.load_state_dict(state_dict)
# classifier.eval()  #设置为评估状态
experiment_dir = 'log/classification/pointnet2_msg_normals/'
num_class = 40
model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
model = importlib.import_module(model_name)
classifier = model.get_model(num_class)
classifier = classifier.cuda()

checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])
# 点云转置

points=torch.from_numpy(points)
print(points.shape)
point = points.transpose(1, 0).contiguous()
print('point.transpose(1, 0).shape: ',point.shape)

point = Variable(point.view(1, point.size()[0], point.size()[1]))  #转为torch变量1,3,2500
print('--------------------')

print(point.dtype)
# point=torch.tensor(point,dtype=torch.float32)
pred, _, _ = classifier(point)   #分割
print(pred)

pred_choice = pred.data.max(2)[1]
print(pred_choice.numpy())   #输出每一个点的预测类别

# print(pred_choice.size())
print(pred_choice.numpy()[0])  #[1 1 1 ... 1 1 1]
pred_color = cmap[pred_choice.numpy()[0], :]   #根据分类结果显示颜色
print('\npred_color: ',pred_color.shape,'\n')
print(pred_color.dtype)

# point_np=point.numpy().reshape(2500,3)
print(point_np.shape)
print(point_np.dtype)
showpoints(point_np, pred_color,pred_color)  #pred_colord的为(2500, 3)的矩阵
