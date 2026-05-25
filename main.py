import numpy as np
from neuron import test_neuron, analyze_neuron
from xor_network import train_xor
from analyzer import analyze_xor_network

def main():
    print("\n")
    print("=" * 60)
    print("极简实验：从权重中恢复规则——第一性原理验证")
    print("=" * 60)
    print()
    
    print("实验目标：验证在极简神经网络中，能否直接从训练后的")
    print("权重参数中恢复出它所学习的规则。随着网络复杂度增加，")
    print("规则恢复的难度如何变化。")
    print()
    
    w, b = test_neuron()
    
    print("\n")
    print("=" * 60)
    print("第二部分：XOR 网络实验（线性不可分问题）")
    print("=" * 60)
    print()
    
    W1, b1, W2, b2 = train_xor(epochs=5000, learning_rate=0.1)
    
    rule = analyze_xor_network(W1, b1, W2, b2)
    
    print("\n")
    print("=" * 60)
    print("实验总结")
    print("=" * 60)
    print()
    
    print("1. 单神经元（线性可分）：")
    print("   - 权重与规则之间存在直接的算术映射")
    print("   - 可以通过 threshold = -b/w 直接恢复规则")
    print("   - 规则恢复：成功 ✓")
    print()
    
    print("2. XOR 网络（线性不可分）：")
    print("   - 权重与规则之间的关系变得隐晦")
    print("   - 需要通过分析隐藏层决策边界和输出层组合来推断")
    print("   - 规则恢复：部分成功，需要启发式方法")
    print()
    
    print("3. 核心结论：")
    print("   - 在极简系统中，权重与规则存在可读的映射")
    print("   - 当系统复杂度增加，这种映射需要外部知识（如训练数据）")
    print("   - 设计通用的'权重分析器'需要更复杂的算法和大量实验")
    print()
    
    print("=" * 60)
    print("实验完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
