import random
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return max(0, z)

def relu_derivative(z):
    return 1 if z > 0 else 0

def initialize_parameters():
    random.seed(42)
    W1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(2)]
    b1 = [random.gauss(0, 0.5) for _ in range(2)]
    W2 = [[random.gauss(0, 0.5) for _ in range(2)]]
    b2 = [random.gauss(0, 0.5)]
    return W1, b1, W2, b2

def dot_product(matrix, vector):
    result = []
    for row in matrix:
        sum_val = 0
        for i in range(len(row)):
            sum_val += row[i] * vector[i]
        result.append(sum_val)
    return result

def forward(X, W1, b1, W2, b2):
    z1 = []
    for i in range(len(W1)):
        sum_val = 0
        for j in range(len(W1[i])):
            sum_val += W1[i][j] * X[j]
        z1.append(sum_val + b1[i])
    
    a1 = [relu(z) for z in z1]
    
    z2 = []
    for i in range(len(W2)):
        sum_val = 0
        for j in range(len(W2[i])):
            sum_val += W2[i][j] * a1[j]
        z2.append(sum_val + b2[i])
    
    a2 = [sigmoid(z) for z in z2]
    
    return z1, a1, z2, a2

def train_xor(epochs=5000, learning_rate=0.1):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    
    W1, b1, W2, b2 = initialize_parameters()
    
    print("训练 XOR 网络...")
    for epoch in range(epochs):
        total_loss = 0
        
        for idx in range(len(X)):
            z1, a1, z2, a2 = forward(X[idx], W1, b1, W2, b2)
            
            dz2 = a2[0] - y[idx]
            dW2 = [dz2 * a1[j] for j in range(2)]
            db2 = [dz2]
            
            da1 = [W2[0][j] * dz2 for j in range(2)]
            dz1 = [da1[j] * relu_derivative(z1[j]) for j in range(2)]
            dW1 = [[dz1[i] * X[idx][j] for j in range(2)] for i in range(2)]
            db1 = dz1[:]
            
            for i in range(2):
                for j in range(2):
                    W1[i][j] -= learning_rate * dW1[i][j]
                b1[i] -= learning_rate * db1[i]
            
            for j in range(2):
                W2[0][j] -= learning_rate * dW2[j]
            b2[0] -= learning_rate * db2[0]
            
            total_loss += (a2[0] - y[idx]) ** 2
        
        if epoch % 1000 == 0:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    print("训练完成！")
    print()
    
    print("训练后的预测结果：")
    print("-" * 30)
    for i in range(len(X)):
        x1, x2 = X[i][0], X[i][1]
        z1, a1, z2, a2 = forward(X[i], W1, b1, W2, b2)
        pred = a2[0]
        print(f"输入: ({x1}, {x2}), 预测: {pred:.4f}, 真实: {y[i]}")
    print()
    
    print("训练后的权重矩阵：")
    print("-" * 30)
    print(f"W1 (隐藏层权重):")
    for row in W1:
        print(f"  [{row[0]:.4f}, {row[1]:.4f}]")
    print(f"b1 (隐藏层偏置): [{b1[0]:.4f}, {b1[1]:.4f}]")
    print(f"W2 (输出层权重): [{W2[0][0]:.4f}, {W2[0][1]:.4f}]")
    print(f"b2 (输出层偏置): {b2[0]:.4f}")
    print()
    
    return W1, b1, W2, b2

if __name__ == "__main__":
    W1, b1, W2, b2 = train_xor()
