# 代码说明

main.py可视化两个三角形

collide_detection.py 两个2D三角形的碰撞检测

physics_gcn.py 用gcn网络做了模型，输入点的位置和边的信息，输出点的位移


目前问题，图神经网络老是输出0？  model_value网络输出一个分数值，我在设计网络的时候在输出前relu了一下，导致小于0的话直接输出0