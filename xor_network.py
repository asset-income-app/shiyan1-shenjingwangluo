import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def initialize_parameters():
    np.random.seed(42)
    W1 = np.random.randn(2, 2) * 0.5
    b1 = np.random.randn(2) * 0.5
    W2 = np.random.randn(1, 2) * 0.5
    b2 = np.random.randn(1) * 0.5
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    z1 = np.dot(W1, X) + b1.reshape(-1, 1)
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2.reshape(-1, 1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def backward(X, y, W1, b1, W2, b2, z1, a1, z2, a2, learning_rate=0.1):
    m = X.shape[1]
    
    dz2 = a2 - y
    dW2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    
    da1 = np.dot(W2.T, dz2)
    dz1 = da1 * relu_derivative(z1)
    dW1 = (1/m) * np.dot(dz1, X.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1.flatten()
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2.flatten()
    
    return W1, b1, W2, b2

def train_xor(epochs=5000, learning_rate=0.1):
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    y = np.array([[0, 1, 1, 0]])
    
    W1, b1, W2, b2 = initialize_parameters()
    
    print("训练 XOR 网络...")
    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)
        W1, b1, W2, b2 = backward(X, y, W1, b1, W2, b2, z1, a1, z2, a2, learning_rate)
        
        if epoch % 1000 == 0:
            loss = np.mean((a2 - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    print("训练完成！")
    print()
    
    z1, a1, z2, a2 = forward(X, W1, b1, W2, b2)
    print("训练后的预测结果：")
    print("-" * 30)
    for i in range(X.shape[1]):
        x1, x2 = X[0, i], X[1, i]
        pred = a2[0, i]
        print(f"输入: ({x1}, {x2}), 预测: {pred:.4f}, 真实: {y[0, i]}")
    print()
    
    print("训练后的权重矩阵：")
    print("-" * 30)
    print(f"W1 (隐藏层权重):\n{W1}")
    print(f"b1 (隐藏层偏置): {b1}")
    print(f"W2 (输出层权重):\n{W2}")
    print(f"b2 (输出层偏置): {b2}")
    print()
    
    return W1, b1, W2, b2

if __name__ == "__main__":
    W1, b1, W2, b2 = train_xor()
