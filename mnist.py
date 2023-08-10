import torch
import torchvision.datasets
import matplotlib.pyplot as plt
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
import cv2
from torch.utils.tensorboard import SummaryWriter

# 使用随机化种子使神经网络的初始化每次都相同
torch.manual_seed(1)

# 超参数
EPOCH = 20  # 训练整批数据的次数
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False
BATCH_SIZE = 64
LR = 0.001  # 学习率

train_data = torchvision.datasets.MNIST(root='dataset', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)

test_data = torchvision.datasets.MNIST(root='dataset', train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

# 批训练 50个samples， 1个channel， 28x28 (50, 1, 28, 28)
# Torch中的DataLoader是用来包装的数据工具，它能够帮我们有效迭代数据，这样可以进行批训练
# shuffle为true一般打乱数据
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# 定义一个转换，将张量转换为PIL图像
to_pil = torchvision.transforms.ToPILImage()

# 查看64个图像拼凑的一张图片
imgs, labels = next(iter(train_dataloader))
img = torchvision.utils.make_grid(imgs) # 把64张图片拼接为一张图片
# pytorch网络输入图像的格式为（C,H,W），而numpy中图像的shaoe为（H，W，C）故需要变换通道才能有效输出
img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
# print(labels)
# plt.imshow(img)
# plt.show()

# writer = SummaryWriter('logs_mnist')

for data in train_dataloader:
    imgs, targets = data
    # print(imgs[0])
    # print(imgs.shape)
    # print(targets[0])
    # # 取出第一张图片
    # pil_image = imgs[0]
    # # 将张量转换为PIL图像
    # pil_image = to_pil(pil_image)
    # pil_image.show()
    break


# writer.close()

# 进行测试
# 为节约时间，测试时只测试前2000个
test_x = torch.unsqueeze(test_data.train_data, dim=1).type(torch.FloatTensor)[:2000] / 255
# torch.unsqueeze(a) 是用来对数据维度进行扩充，这样shape就从(2000,28,28)->(2000,1,28,28)
# 图像的pixel本来是0到255之间，除以255对图像进行归一化使取值范围在(0,1)
test_y = test_data.test_labels[:2000]




# 用class类来建立CNN模型
# CNN流程：卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)->
#        展平多维的卷积成的特征图->接入全连接层(Linear)->输出
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)->激励函数(ReLu)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积cond2d
            nn.Conv2d( # 输入图像太小(1,28,28)
                in_channels=1, # 输入图片是灰度图像只有一个通道
                out_channels=16,
                kernel_size=5, # 卷积核大小
                stride=1,
                padding=2 # 想要cond2d输出的图片长宽不变就要进行补0操作 padding = (kernel_size-1)/2
            ), # 输出图像大小(16, 28, 28)
            # 激活函数
            nn.ReLU(),
            # 池化 下采样
            nn.MaxPool2d(kernel_size=2) # 在2x2空间下采样
            # 输出图像大小(16,14,14)
        )

        # 建立第二个卷积(Conv2d)->激励函数(ReLu)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ), # 输出图像大小(32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # 输出图像大小(32,7,7)
        )

        # 建立全卷积连接层
        self.out = nn.Linear(32 * 7 * 7, 10) # 输出是十个类

    # 下面定义x的传播路线
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        output = self.out(x)
        return output

# device =torch.device('cuda:0')
cnn = CNN()
# print(cnn)

# 训练
# 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差

# 优化器选择Adam
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 损失函数
loss_func = nn.CrossEntropyLoss() # 目标标签是one-hotted

# 开始训练
# for epoch in range(EPOCH):
#     for step, (b_x, b_y) in enumerate(train_dataloader):
#         output = cnn(b_x) # 先将数据放到cnn中计算output
#         loss = loss_func(output, b_y) # 输出和真实标签的loss，二者位置不可以颠倒
#         optimizer.zero_grad() # 清楚之前学到的梯度的参数
#         loss.backward() # 反向传播，计算梯度
#         optimizer.step() # 应用梯度
#
#         if step % 64 ==0:
#             test_output = cnn(test_x)
#             # print(test_output)
#
#             pred_y = torch.max(test_output, 1)[1].data.numpy()
#             accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
#             print('Epoch', epoch+1, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
#
#
# # 保存模型
# torch.save(cnn.state_dict(), 'cnn.pkl')


# 加载模型，调用时需将前面训练及保存模型的代码注释掉，否则会再训练一遍
cnn.load_state_dict(torch.load('cnn.pkl', map_location=torch.device('cpu')))
cnn.eval()
# print 10 predictions from test data
inputs = test_x[:32]  # 测试32个数据
test_output = cnn(inputs)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')  # 打印识别后的数字
# print(test_y[:10].numpy(), 'real number')

img = torchvision.utils.make_grid(inputs)
img = img.numpy().transpose(1, 2, 0)

# 下面三行为改变图片的亮度
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img * std + mean
cv2.imshow('win', img)  # opencv显示需要识别的数据图片
key_pressed = cv2.waitKey(0)