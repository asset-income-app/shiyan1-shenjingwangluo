import math

def analyze_neuron(w, b):
    threshold = -b / w
    if w > 0:
        rule = f"规则：如果 x > {threshold:.4f}，则 y=1"
    else:
        rule = f"规则：如果 x < {threshold:.4f}，则 y=1"
    return threshold, rule

def relu(z):
    return max(0, z)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

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

def analyze_xor_network(W1, b1, W2, b2):
    print("=" * 50)
    print("XOR 网络权重分析")
    print("=" * 50)
    print()
    
    print("步骤1：分析隐藏层神经元的决策边界")
    print("-" * 40)
    for i in range(len(W1)):
        w11, w12 = W1[i][0], W1[i][1]
        bi = b1[i]
        
        if abs(w12) > 1e-6:
            slope = -w11 / w12
            intercept = -bi / w12
            print(f"隐藏层神经元 {i+1}:")
            print(f"  权重: w11={w11:.4f}, w12={w12:.4f}, b={bi:.4f}")
            print(f"  决策边界: x2 = {slope:.4f} * x1 + {intercept:.4f}")
        else:
            print(f"隐藏层神经元 {i+1}:")
            print(f"  权重: w11={w11:.4f}, w12={w12:.4f}, b={bi:.4f}")
            print(f"  决策边界: x1 = {-bi/w11:.4f}")
    print()
    
    print("步骤2：测试隐藏层在四个输入点的激活情况")
    print("-" * 40)
    test_points = [[0, 0], [0, 1], [1, 0], [1, 1]]
    
    for i in range(len(test_points)):
        x1, x2 = test_points[i][0], test_points[i][1]
        z1, a1, z2, a2 = forward(test_points[i], W1, b1, W2, b2)
        print(f"输入 ({x1}, {x2}):")
        for j in range(len(a1)):
            print(f"  隐藏层神经元 {j+1} 激活值: {a1[j]:.4f}")
    print()
    
    print("步骤3：分析输出层的组合方式")
    print("-" * 40)
    print(f"输出层权重: W2 = [{W2[0][0]:.4f}, {W2[0][1]:.4f}]")
    print(f"输出层偏置: b2 = {b2[0]:.4f}")
    print()
    
    print("步骤4：推断规则")
    print("-" * 40)
    
    output_patterns = []
    for i in range(len(test_points)):
        x1, x2 = test_points[i][0], test_points[i][1]
        z1, a1, z2, a2 = forward(test_points[i], W1, b1, W2, b2)
        output_patterns.append({
            'input': (x1, x2),
            'hidden': a1[:],
            'output': a2[0]
        })
    
    positive_cases = [p for p in output_patterns if p['output'] > 0.5]
    negative_cases = [p for p in output_patterns if p['output'] <= 0.5]
    
    print("输出为 1 的情况：")
    for p in positive_cases:
        print(f"  输入 {p['input']}, 隐藏层激活: [{p['hidden'][0]:.2f}, {p['hidden'][1]:.2f}]")
    
    print("\n输出为 0 的情况：")
    for p in negative_cases:
        print(f"  输入 {p['input']}, 隐藏层激活: [{p['hidden'][0]:.2f}, {p['hidden'][1]:.2f}]")
    print()
    
    rule = infer_rule(output_patterns)
    print(f"推断的规则：{rule}")
    print()
    
    return rule

def infer_rule(output_patterns):
    positive_inputs = [p['input'] for p in output_patterns if p['output'] > 0.5]
    negative_inputs = [p['input'] for p in output_patterns if p['output'] <= 0.5]
    
    if len(positive_inputs) == 2 and len(negative_inputs) == 2:
        pos_set = set(positive_inputs)
        neg_set = set(negative_inputs)
        
        if pos_set == {(0, 1), (1, 0)} and neg_set == {(0, 0), (1, 1)}:
            return "异或规则：当两个输入不同时输出 1，相同时输出 0"
        elif pos_set == {(0, 0), (1, 1)} and neg_set == {(0, 1), (1, 0)}:
            return "同或规则：当两个输入相同时输出 1，不同时输出 0"
    
    rule_parts = []
    for p in output_patterns:
        if p['output'] > 0.5:
            x1, x2 = p['input']
            rule_parts.append(f"({x1}=={x1} and {x2}=={x2})")
    
    if rule_parts:
        return f"当 {' 或 '.join(rule_parts)} 时输出 1"
    else:
        return "无法从权重中直接推断出明确的规则"

if __name__ == "__main__":
    W1 = [[1.0, 1.0], [1.0, 1.0]]
    b1 = [-0.5, -1.5]
    W2 = [[1.0, -1.0]]
    b2 = [0.0]
    
    rule = analyze_xor_network(W1, b1, W2, b2)
    print(f"\n最终推断的规则：{rule}")
