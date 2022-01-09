import poemv2
import numpy as np
import torch

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
LR = 0.001
MAX_GEN_LEN = 48
EPOCHS = 20

#预处理部分
sample = np.load("D://AI computer system Lab/MyLab/DLcourse/Lab3/tang.npz",
                     allow_pickle=True)
ix2word = sample['ix2word'].item()
word2ix = sample['word2ix'].item()
# 类定义模型
model = poemv2.PoetryModel(len(ix2word), EMBEDDING_DIM, HIDDEN_DIM)
# 载入参数
model.load_state_dict(torch.load("D://AI computer system Lab/MyLab/DLcourse/Lab3/tang_final.pth"))
print("Please input the first half sentence of poem!")
# 输入诗句
first = str(input())
#生成诗句
ge_po = poemv2.generate(model, first, ix2word, word2ix)
gen_poetry = ''.join(ge_po)
print("The predicting poem is as follows ：" ,gen_poetry)
print("Thanks for your using and Pls give 5-star appreciation")