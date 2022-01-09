import trainSQ
from torch import nn
from torch.utils.data import DataLoader
import torch
import averagevaluemeter
from matplotlib import pyplot as plt 
import numpy as np

train_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/train.txt"
test_path  = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/test.txt"
validation_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/validation.txt"
pred_word2vec_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/wiki_word2vec_50.bin"
model_save_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/modelDict/model.pth"
embedding_dim = 50
hidden_dim = 100
batch_size = 16
LSTM_layers = 3
drop_prob = 0.5
lr = 0.001
epochs = 4
vocab_size = 2

def test(test_loader, model, criterion):
    test_acc = 0.0
    test_acc_box, test_iter_box = [], []

    model.eval()
    with torch.no_grad():  
        test_top1 = averagevaluemeter.AverageValueMeter()
        for i, data in enumerate(test_loader): 
            inputs = data[0]
            labels = data[1]
            batchs = data[2] 

            results, hidden = model(inputs, batchs)
            loss = criterion(results, labels)
# 计算当前iter正确率
            prec1, _ = trainSQ.accuracy(results, labels, topk=(1, 2))
            n = inputs.size(0)
# 计算总的正确率
            test_top1.add(prec1.item(), n)
# 计算损失
            #test_loss += loss.item()
# 存储
            #test_loss_box.append(test_loss / (i + 1))
            test_acc_box.append(test_top1.avg)
# 显示
            print('Test_acc is ', '%.6f' % test_top1.avg)
# 最终正确率
        test_acc = test_top1.avg
    return test_acc, test_acc_box, test_iter_box

word2ix, ix2word = trainSQ.train_dict(train_path)

test_data = trainSQ.mydataset(test_path, word2ix, ix2word)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False,num_workers=0, collate_fn=trainSQ.Fun_predata,)
weight = torch.zeros(len(word2ix), embedding_dim)
model = trainSQ.PoetryModel(embedding_dim, vocab_size,hidden_dim,LSTM_layers,drop_prob,weight)
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load(model_save_path))  # 模型加载
    
test_acc, test_acc_box, test_iter_box=test(test_loader, model, criterion)
print("Over!")

plt.figure(1)   
plt.plot(np.arange(len(test_loader)),test_acc_box)
plt.title("Test acc--Test iter") 
plt.xlabel("Test iter") 
plt.ylabel("Test acc")
plt.legend()
plt.show()