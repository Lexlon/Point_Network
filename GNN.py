import torch
import numpy as np
import dgl
from dgl.nn.pytorch import GraphConv

# 生成20个城市点的坐标，范围为0-100
n = 20  # 城市数量
coords = np.random.randint(0, 100, size=(n, 2))

# 计算每两个城市之间的距离矩阵
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i][j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

# 构建完全图，每条边的权重为对应城市之间的距离
g = dgl.complete_graph(n)  # 完全图有n*(n-1)/2条边
g.edata['w'] = torch.from_numpy(dist_matrix[g.edges()[0], g.edges()[1]]).float()  # 边权重


# 定义GNN模型，使用GraphConv层
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h


# 实例化模型，输入维度为2（坐标），隐藏维度为16，输出维度为1（得分）
model = GNN(2, 16, 1)

# 定义优化器和损失函数，使用Adam优化器和负对数似然损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss()

# 将坐标转换为张量作为节点特征
x = torch.from_numpy(coords).float()

# 训练模型，迭代次数为1000
epochs = 1000
for epoch in range(epochs):
    # 前向传播，得到每个节点的得分
    logits = model(g, x).squeeze()

    # 将得分归一化为概率分布，使用gumbel softmax采样一个路径作为预测值
    probs = torch.softmax(logits / 0.2, dim=0)  # 温度参数设为0.2
    pred_path = torch.argmax(torch.nn.functional.gumbel_softmax(probs.log(), tau=0.2), dim=0)

    # 计算预测路径的总距离作为损失值，并反向传播更新参数
    pred_dist = g.edata['w'][g.edge_id(pred_path[:-1], pred_path[1:])].sum()
    loss = pred_dist
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100次迭代打印一次损失值和预测路径
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss {loss.item():.4f}")
        print(f"Predicted path: {pred_path.tolist()}")

# bing 写的代码