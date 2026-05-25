import numpy as np

def step_function(z):
    return 1 if z >= 0 else 0

def neuron_forward(x, w, b):
    z = w * x + b
    return step_function(z)

def analyze_neuron(w, b):
    threshold = -b / w
    if w > 0:
        rule = f"规则：如果 x > {threshold:.4f}，则 y=1"
    else:
        rule = f"规则：如果 x < {threshold:.4f}，则 y=1"
    return threshold, rule

def test_neuron():
    w = 1.0
    b = -0.5
    
    print("=" * 50)
    print("单神经元实验：线性可分问题")
    print("=" * 50)
    print(f"权重 w = {w}")
    print(f"偏置 b = {b}")
    print()
    
    threshold, rule = analyze_neuron(w, b)
    print(rule)
    print(f"计算得到的阈值：{threshold:.4f}")
    print()
    
    test_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    print("测试结果：")
    print("-" * 30)
    for x in test_values:
        y = neuron_forward(x, w, b)
        print(f"x = {x:.1f} -> y = {y}")
    print()
    
    expected_rule = "规则：如果 x > 0.5，则 y=1"
    print(f"目标规则：{expected_rule}")
    print(f"分析规则：{rule}")
    if abs(threshold - 0.5) < 0.0001:
        print("✓ 规则恢复成功！")
    else:
        print("✗ 规则恢复失败！")
    print()
    
    return w, b

if __name__ == "__main__":
    test_neuron()
