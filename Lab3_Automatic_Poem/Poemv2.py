from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision
!pip install torchnet

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchnet import meter

EMBEDDING_DIM = 256
HIDDEN_DIM = 1024
LR = 0.001
MAX_GEN_LEN = 200
EPOCHS = 20
DROP_PROB = 0.5
LSTM_LAYER = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Part 1、数据准备
def prepareData():
    datas = np.load("drive/My Drive/Notebooks/tang.npz", allow_pickle=True)
    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data,batch_size=16,shuffle=True,num_workers=0)
    return dataloader, ix2word, word2ix

#测试
poem_loader, ix2word, word2ix = prepareData()
#print(word2ix['哪'])
#print(ix2word) # {0: '<character>', ... , 8290 : '<EOP>', 8291: '<START>', 8292: '</s>'}
#print(word2ix['<EOP>']) # 8290
num_box = 0
#for li, oth in enumerate(poem_loader):
#    num_box += 1
#print(num_box)


# Part 2、模型构建
class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, drop_prob, lstm_layers):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=self.lstm_layers, batch_first=True)
        #self.linear = nn.Linear(self.hidden_dim, vocab_size)
        self.linear1 = nn.Linear(self.hidden_dim,2048)
        self.linear2 = nn.Linear(2048,4096)
        self.linear3 = nn.Linear(4096,vocab_size)
        #增加dropout
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input, hidden=None):
        embeds = self.embeddings(input)
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(self.lstm_layers, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(self.lstm_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        sample_pre, hidden = self.lstm(embeds, (h_0, c_0))
        # 多加两层
        sample_pre = torch.tanh(self.linear1(sample_pre))
        #sample_pre = self.dropout(sample_pre)
        sample_pre = torch.tanh(self.linear2(sample_pre))
        sample_pre = self.linear3(sample_pre)
        sample_pre = sample_pre.reshape(batch_size * seq_len, -1)
        return sample_pre, hidden


# Part 3、模型训练
def train(epochs, poem_loader, word2ix):
    # 定义模型、设置优化器和损失函数、获取模型输出、计算误差、误差反向传播等步骤
    model = PoetryModel(len(word2ix),embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, drop_prob=DROP_PROB,lstm_layers=LSTM_LAYER) 
    model.train()
    model.to(device) # 移动模型到cuda
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10,gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    #model.load_state_dict(torch.load("drive/My Drive/Notebooks/tang.pth"))
    loss_meter = meter.AverageValueMeter()

    loss_box = []
    iter_box = []

    for epoch in np.arange(epochs):
        loss_meter.reset()
        for chara_index, character in enumerate(poem_loader):
            print("Epoch ",epoch," : It has trained for ",chara_index," iterations!")
            character = character.long().transpose(1,0).contiguous()
            character = character.to(device)
            #character.contiguous()
            #sample = character[0].to(device)
            #target = character[1].to(device)
            #target = target.view(-1)
            optimizer.zero_grad()
            sample = character[:-1, :]
            target = character[1:, :]
            sample_pre, _ = model(sample)
            loss = criterion(sample_pre,target.view(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
        # 输出内容即可
        torch.save(model.state_dict(),'drive/My Drive/Notebooks/tang.pth')
        print("It has saved .pth file for ",epoch," times!") 
        model.load_state_dict(torch.load("drive/My Drive/Notebooks/tang.pth"))
        loss_box.append(loss)
        iter_box.append(epoch)
        scheduler.step()     
    print("Finish all!") 

# Part 4、模型预测
def generate(model, start_words, ix2word, word2ix):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.cuda()
    hidden = None
    #model.eval()
    with torch.no_grad():
        for i in range(MAX_GEN_LEN):
            sample_pre, hidden = model(input, hidden)
		# 如果在给定的句首中，input为句首中的下一个字
            if i < start_words_len:
               w = results[i]
               input = input.data.new([word2ix[w]]).view(1, 1)
               # 否则将output作为下一个input进行
            else:
               top_index = sample_pre.data[0].topk(1)[1][0].item()
               w = ix2word[top_index]
               results.append(w)
               input = input.data.new([top_index]).view(1, 1)
            if w == '<EOP>':
               del results[-1]
               break
    return results

if __name__ == '__main__':
    train(EPOCHS, poem_loader, word2ix)