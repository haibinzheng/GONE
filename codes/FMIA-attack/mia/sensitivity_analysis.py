from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
x = [0.001, 0.003, 0.005, 0.01]
MNIST = [0.001, 0.003, 0.005, 0.01]
CIFAR10 = [0.4978, 0.5043, 0.5303, 0.798]
CIFAR100 =  [0.001, 0.003, 0.005, 0.01]
Face = [0.4982, 0.5036, 0.5223, 0.7657]
News =  [0.001, 0.003, 0.005, 0.01]
Location = [0.5005, 0.5985, 0.7085,0.8587]
Purchase2 = []
Purchase10 = []
Purchase20 = []
Purchase50 = []
Purchase100 = []

# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_acc, y_acc, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='accuracy')
plt.plot(x_pre, y_pre, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='precision')
plt.plot(x_recall, y_recall, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='recall')

# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
plt.xlabel('Epi')
# plt.ylabel('y轴数字')

plt.show()
# plt.savefig('demo.jpg')  # 保存该图片
