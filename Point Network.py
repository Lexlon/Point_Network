import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 随机生成30个城市的坐标
num_nodes = 30
coord = np.random.rand(num_nodes, 2) * 100

# 计算距离矩阵
dist_matrix = tf.sqrt(tf.reduce_sum(tf.square(coord[:, None, :] - coord[None, :, :]), axis=-1))

# 定义模型
class PointNetwork(tf.keras.Model):
    def __init__(self, in_feats, out_feats):
        super(PointNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(out_feats)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x

# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)

# 训练模型
@tf.function
def train_step(model, inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.reduce_mean(tf.square(predictions - labels))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model, inputs, labels, epochs=1000):
    for epoch in range(epochs):
        loss = train_step(model, inputs, labels)
        if epoch % 100 == 0:
            print('Epoch {:d} | Loss {:.4f}'.format(epoch, loss.numpy()))

# 定义训练数据
x_train = coord.astype(np.float32)
y_train = dist_matrix.numpy().astype(np.float32)

# 创建模型实例并训练
model = PointNetwork(2, num_nodes)
train(model, x_train, y_train)

# 预测最短路径
x_test = tf
# 预测最短路径
x_test = tf.convert_to_tensor(coord, dtype=tf.float32)
y_pred = model(x_test)
path = np.arange(num_nodes)
path_order = np.argsort(y_pred.numpy(), axis=-1)[:, path]
best_path = path_order[0]
print('Best path:', best_path)

# 可视化结果
plt.figure(figsize=(6, 6))
plt.plot(coord[:, 0], coord[:, 1], 'o', markersize=10)
plt.plot(coord[best_path, 0], coord[best_path, 1], 'r-')
plt.axis('equal')
plt.show()
