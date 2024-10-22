import numpy as np
from scipy.special import expit

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):#学习率和样本数量
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):#逻辑回归函数
        return expit(z)

    def fit(self, X, y):#训练模型函数
        # 样本数量和特征数量
        n_samples, n_features = X.shape#X是一个矩阵

        # 初始化权重和偏置项
        self.weights = np.zeros(n_features)#w
        self.bias = 0#b
        lam=0.1#正规化强度
        # 梯度下降训练
        for i in range(self.num_iterations):#进行num_iterations次梯度下降
            # 计算线性模型
            linear_model = np.dot(X, self.weights) + self.bias
            # 应用 Sigmoid 激活函数
            y_predicted = self.sigmoid(linear_model)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))+(lam/n_samples)*self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 更新权重和偏置项
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # 计算预测概率
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)#这里的y_predicted是一个列表，python中的函数可以直接处理列表并返回一个列表
        # 将概率值转化为二分类结果
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)


# 使用示例
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # 使用sklearn的乳腺癌数据集
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 初始化和训练模型
    model = LogisticRegression(learning_rate=0.001, num_iterations=1000)
    model.fit(X_train, y_train)

    # 预测结果
    predictions = model.predict(X_test)

    # 计算准确率
    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy:.4f}")
