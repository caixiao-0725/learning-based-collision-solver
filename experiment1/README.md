#实验1记录

目标：给定两个固定位置的已经相交的三角形，尝试在一定的步长内把它俩分开

两种想法：1.给每个action一个最大的移动距离，这个可以用softmax实现  main2.py
         
         2.在reward函数里面加入action的模长，来限制action的大小   main1.py

main3.py  尝试减小main2.py中的maxstep

在使用方法2的时候，前面几个训练epoch会产生非常小的reward，-1e9的数量级都有，输出action以后发现，action非常大，尝试多训练一会儿看看能不能work。



##实验过程中的疑问和解决

q1:图神经网络老是输出0？  

model_value网络输出一个分数值，我在设计网络的时候在输出前relu了一下，导致小于0的话直接输出0


q2:三个点直接聚拢在一个点了！咋整?

加随机数，但是直接寄了，每个点都跑到很远的地方,不知道为啥reward收敛不了

q3:也许我应该只记录赢的时候的数据，输的时候的数据是否具有指导意义？

使用Self-imitation learning


##吐槽

勾吧保存了模型没保存求解器，难怪每次断开读取模型接着训练就寄了  
DONE

也许我需要修改一下探索部分的程序 不应该每次都直接加个噪声，而应该用个random一定概率的情况下使用噪声             
DONE




##结果
main2.py已经可以做到将两个三角形一步分开了，不过我会在main3.py里面逐步缩小max step size

