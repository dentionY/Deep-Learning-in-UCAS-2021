import torch
import re 
from torch.utils.data import Dataset, DataLoader
from zhconv import convert  
import gensim 
from torch import nn
import averagevaluemeter
import torch.optim as optim
from matplotlib import pyplot as plt 
import numpy as np

train_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/train.txt"
test_path  = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/test.txt"
validation_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/validation.txt"
wiki_word2vec_50_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/Dataset/wiki_word2vec_50.bin"
model_save_path = "D:/AI computer system Lab/MyLab/DLcourse/Lab4/modelDict/model.pth"
embedding_dim = 50
hidden_dim = 100
batch_size = 16
LSTM_layers = 3
drop_prob = 0.5
lr = 0.001
epochs = 4
vocab_size = 2

# 查看三个txt文档，会发现有拼音、英文、繁体汉字、简体汉字
# train_dict()专用于train.txt文件，提取其词典
def train_dict(train_path):
    words = []
    word2ix = {}
    ix2word = {}

    # UnicodeDecodeError: 'gbk' codec can't decode byte 0xb1 in position 11: illegal multibyte sequence
    # 增加 -- > encoding='utf-8'
    # https://blog.csdn.net/marselha/article/details/91872832
    with open(train_path,'r',encoding='UTF-8') as f:
        lines = f.readlines()
        for line in  lines:
            line = convert(line, 'zh-cn')
            line_words = re.split(r'[\s]', line)
            # 第一个元素是label
            li_words = line_words[1:-1]
            for w in li_words:
                words.append(w)
    words = list(set(words))
    word_index = sorted(words) 

    for index, word in enumerate(word_index):
        word2ix_tmp = {word : index+1}
        word2ix.update(word2ix_tmp)
        ix2word_tmp = {index+1 : word}
        ix2word.update(ix2word_tmp)

# 添加这一部分：测试集和验证集上有，训练集上无
    word2ix['CanNotFound'] = 0
    ix2word[0] = 'CanNotFound'
    return word2ix, ix2word

word2ix, ix2word = train_dict(train_path)
#print("word2ix is ",word2ix)
#print("ix2word is ",ix2word)
# 51405个元素

# Part 2 dataset.py
def Fun_predata(data):

    length_tmp_box = []
    #length_np_box = []
    for i in data:
        i_tmp = (len(i[0]),i[1])
        length_tmp_box.append(i_tmp)
    length_tmp_box.sort(key = lambda x :x[0],reverse=True)
    new_data = []
    for i in length_tmp_box:
        for j in data:
            if i[1] == j[1] and i[0] == len(j[0]):
                new_data_tmp = (j[0],j[1])
                new_data.append(new_data_tmp)

    datalength = []
    for i in new_data:
        datalength.append(len(i[0]))

    txt_data = []
    label_data = []
    for i in new_data:
        txt_data.append(i[0])
        label_data.append(i[1])
    
    txt_data = torch.nn.utils.rnn.pad_sequence(sequences=txt_data, batch_first=True, padding_value=0)
    #返回[batch_size, M],M为batch中的最大长度
    label_data = torch.tensor(label_data)
    return txt_data, label_data, datalength


class mydataset(Dataset):
    def __init__(self, dir, word2ix, ix2word):
        self.dir = dir
        self.word2ix = word2ix
        self.ix2word = ix2word
        self.data, self.label = self.get_data_label()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx]

    def get_data_label(self):
        data = []
        label = []
        with open(self.dir, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            # 先检测每一行的起始元素是否为1或者0.如果不是，直接跳过。
            # Tips ：train.txt 看到第6711行开头的不是标签，所以必须处理这个问题。无法排除另外两个文件是否存在同样问题
            # Solution : 以上问题可借助try ... except ... 解决
            for i in lines:
                try:
                    detect = torch.tensor(int(i[0]), dtype=torch.int64)
                    label.append(detect)
                except BaseException:  
                    continue
                jianti_i = convert(i, 'zh-cn') 
                jianti_words_labels = re.split(r'[\s]', jianti_i)  
                jianti_words = jianti_words_labels[1:-1]
                words_to_idx = []
                # Tips : 在验证集和测试集上有word，但在train的word中没有
                # Solution : 以上问题可借助try ... except ... 解决
                for w in jianti_words:
                    try:
                        index = self.word2ix[w]
                    except BaseException:
                        index = 0  # 对应train_dict()
                    words_to_idx.append(index)
                data.append(torch.tensor(words_to_idx, dtype=torch.int64))
        return data, label


# 构建第一层的词嵌入层
size = len(word2ix)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(wiki_word2vec_50_path, binary=True)
# 初始权重定义
weight = torch.zeros(size,embedding_dim)
for i in range(len(word2vec_model.index_to_key)):
    try:
        new_key = word2vec_model.index_to_key[i]
        new_index = word2ix[new_key]
    except:
        continue
    new_word = ix2word[new_index]
    weight[new_index, :] = torch.from_numpy(word2vec_model.get_vector(new_word))

# 紧接着第二层及其以上
# 这里部分参考  自动写诗 的程序框架
class PoetryModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim,LSTM_layers,drop_prob,weight):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(weight)
        # 下面这一行相当重要！！！！
        self.embeddings.weight.requires_grad = True
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=LSTM_layers,batch_first=True, dropout=drop_prob)
    
        self.linear1 = nn.Linear(self.hidden_dim, 2048)
        self.linear2 = nn.Linear(2048, 256)
        self.linear3 = nn.Linear(256,  32)
        #self.linear4 = nn.Linear(256, 32)
        self.linear4 = nn.Linear(32, vocab_size)  

        #增加dropout
        self.dropout = nn.Dropout(drop_prob)


    def forward(self, input, batchs, hidden=None):
        embeds = self.embeddings(input)  
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(LSTM_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(LSTM_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        sample_pre, hidden = self.lstm(embeds, (h_0, c_0)) 
        #多加两层 
        sample_pre = torch.tanh(self.linear1(sample_pre))
        #sample_pre = self.dropout(sample_pre)
        sample_pre = torch.tanh(self.linear2(sample_pre))
        sample_pre = self.dropout(sample_pre)
        sample_pre = torch.tanh(self.linear3(sample_pre))

       # sample_pre = torch.tanh(self.linear4(sample_pre))
       # sample_pre = self.dropout(sample_pre)

        sample_pre = self.linear4(sample_pre)

        sample_out = torch.zeros((sample_pre.shape[0], sample_pre.shape[2]))
        for i in range(len(batchs)):
            sample_out[i] = sample_pre[i][batchs[i] - 1]
        
        return sample_out, hidden

#Reference : https://blog.csdn.net/weixin_28812983/article/details/113964421
def accuracy(output, target, topk=(1,)):
    maxk=max(topk)
    batch_size= target.size(0)
    
    _, pred= output.topk(maxk, 1, True, True) # _, pred = logit.topk(maxk, 1, True, True)
    pred=pred.t()
    correct= pred.eq(target.view(1, -1).expand_as(pred))
    res=[]
    for k in topk:
        #RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        #Solution: https://blog.csdn.net/tiao_god/article/details/108189879
        # add : contiguous()
        correct_k= correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 /batch_size)) # it seems this is a bug, when not all batch has same size, the mean of accuracy of each batch is not the mean of accu of all datasetreturnres
    return res


def train(train_loader,model,criterion,optimizer,scheduler):
    train_loss_box = []
    train_acc_box =[]
    train_iter_box = []
    model.train()
    top1 = averagevaluemeter.AverageValueMeter()
    train_loss = 0.0
    for i, data in enumerate(train_loader): 
        # 先将loader内部的数据分开，data--label--batch
        inputs = data[0]
        labels = data[1]
        batchs = data[2]
        optimizer.zero_grad()
        results, _ = model(inputs, batchs)
        loss = criterion(results, labels)
        loss.backward()
        optimizer.step()
# 计算当前iter正确率
        prec1, _ = accuracy(results, labels, topk=(1, 2))
        n = inputs.size(0)
# 计算总的正确率
        top1.add(prec1.item(), n)
# 计算损失
        train_loss += loss.item()
# 存储
        train_loss_box.append(train_loss / (i + 1))
        train_acc_box.append(top1.avg)
        #train_iter_box.append(i)
# 显示
        print('The line is ',i, ', and remaining line is ',len(train_loader)-i,',train_loss : ', '%.6f' % (train_loss / (i + 1)), 'train_acc : ', '%.6f' % top1.avg)
    scheduler.step()
    return train_loss_box,train_acc_box,train_iter_box


def validate(validate_loader,model,criterion):
    val_acc = 0.0
    valid_loss_box = []
    valid_acc_box = []
    valid_iter_box = []
    model.eval()
    with torch.no_grad():  
        val_top1 = averagevaluemeter.AverageValueMeter()
        validate_loss = 0.0
        for i, data in enumerate(validate_loader): 
            inputs = data[0]
            labels = data[1]
            batchs = data[2] 

            results, _ = model(inputs, batchs)
            loss = criterion(results, labels)
# 计算当前iter正确率
            prec1, _ = accuracy(results, labels, topk=(1, 2))
            n = inputs.size(0)
# 计算总的正确率
            val_top1.add(prec1.item(), n)
# 计算损失
            validate_loss += loss.item()
# 存储
            valid_loss_box.append(validate_loss / (i + 1))
            valid_acc_box.append(val_top1.avg)
            #valid_iter_box.append(i)
# 显示
            print('The line is ',i,', and remaining line is ',len(validate_loader)-i,', validate_loss : ' ,(validate_loss / (i + 1)), 'validate_acc : ', val_top1.avg)
# 最终正确率
        val_acc = val_top1.avg
    return val_acc, valid_loss_box, valid_acc_box, valid_iter_box


# Part 6 main.py
def main():
    # 存储
    whole_train_loss_box = []
    whole_train_acc_box = []
    whole_train_iter_box = []

    whole_val_loss_box = []
    whole_val_acc_box = []
    whole_val_iter_box = []
# 提取数据
    train_data = mydataset(train_path, word2ix, ix2word)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True,num_workers=0, collate_fn=Fun_predata)
    validation_data = mydataset(validation_path, word2ix, ix2word)
    validation_loader = DataLoader(validation_data,batch_size=16,shuffle=True,num_workers=0,collate_fn=Fun_predata)
# 训练模型
    model = PoetryModel(embedding_dim,vocab_size,hidden_dim,LSTM_layers,drop_prob,weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("The epoch is ",epoch)
        if epoch is 0:
            train0_loss_box,train0_acc_box,train0_iter_box = train(train_loader,model,criterion,optimizer,scheduler)
            val0_acc, valid0_loss_box, valid0_acc_box, valid0_iter_box = validate(validation_loader,model,criterion)
        elif epoch is 1 :
            train1_loss_box,train1_acc_box,train1_iter_box = train(train_loader,model,criterion,optimizer,scheduler)
            val1_acc, valid1_loss_box, valid1_acc_box, valid1_iter_box = validate(validation_loader,model,criterion)
        elif epoch is 2 :
            train2_loss_box,train2_acc_box,train2_iter_box = train(train_loader,model,criterion,optimizer,scheduler)
            val2_acc, valid2_loss_box, valid2_acc_box, valid2_iter_box = validate(validation_loader,model,criterion)
        elif epoch is 3 :
            train3_loss_box,train3_acc_box,train3_iter_box = train(train_loader,model,criterion,optimizer,scheduler)
            val3_acc, valid3_loss_box, valid3_acc_box, valid3_iter_box = validate(validation_loader,model,criterion)

    plt.figure(1)
    plt.subplot(2,2,1) 
    plt.plot(np.arange(len(train_loader)),train0_loss_box,label = 'round0')
    plt.plot(np.arange(len(train_loader)),train1_loss_box,label = 'round1')
    plt.plot(np.arange(len(train_loader)),train2_loss_box,label = 'round2')
    plt.plot(np.arange(len(train_loader)),train3_loss_box,label = 'round3')
   
    plt.title("Training loss--Training iter") 
    plt.xlabel("Training iter") 
    plt.ylabel("Training loss")
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(np.arange(len(train_loader)),train0_acc_box,label = 'round0')
    plt.plot(np.arange(len(train_loader)),train1_acc_box,label = 'round1')
    plt.plot(np.arange(len(train_loader)),train2_acc_box,label = 'round2')
    plt.plot(np.arange(len(train_loader)),train3_acc_box,label = 'round3')

    plt.title("Training acc--Training iter") 
    plt.xlabel("Training iter") 
    plt.ylabel("Training acc")
    plt.legend()   

    plt.subplot(2,2,3)
    plt.plot(np.arange(len(validation_loader)),valid0_loss_box,label = 'round0')
    plt.plot(np.arange(len(validation_loader)),valid1_loss_box,label = 'round1')
    plt.plot(np.arange(len(validation_loader)),valid2_loss_box,label = 'round2')
    plt.plot(np.arange(len(validation_loader)),valid3_loss_box,label = 'round3')

    plt.title("Valid loss--Training iter") 
    plt.xlabel("Training iter") 
    plt.ylabel("Valid loss")
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(np.arange(len(validation_loader)),valid0_acc_box,label = 'round0')
    plt.plot(np.arange(len(validation_loader)),valid1_acc_box,label = 'round1')
    plt.plot(np.arange(len(validation_loader)),valid2_acc_box,label = 'round2')
    plt.plot(np.arange(len(validation_loader)),valid3_acc_box,label = 'round3')

    plt.title("Valid acc--Training iter") 
    plt.xlabel("Training iter") 
    plt.ylabel("Valid acc")
    plt.legend()
    plt.show() 
    
    torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    main()