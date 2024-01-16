import data_process
import torch
import m_model
import torch.nn as nn
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def m_train():
    # define the parameter
    mode = 96 # or 336
    batch_size = 64
    epochs = 200
    embedding_dim = 7
    dropout_rate = 0.2
    hidden_dim = 7
    learning_rate = 1e-4

    data_train,data_valid,data_test,train_date,valid_date,test_date = data_process.read_data()
    train_datas,train_labels = data_process.generate_data(mode,data_train)
    valid_datas,valid_labels = data_process.generate_data(mode,data_valid)
    test_datas,test_labels = data_process.generate_data(mode,data_test)

    train_loader,valid_loader,test_loader = data_process.get_loader(train_datas,train_labels,
                                                                    valid_datas,valid_labels,
                                                                    test_datas,test_labels,batch_size)



    model = m_model.model_lstm(embedding_dim,hidden_dim,dropout_rate).cuda()
    loss_MSE = nn.L1Loss(reduce=True,size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    iters = 0
    no_improve = 0
    best_loss = float('inf')
    print('\n---------tarin loader: {}, validation loader: {}, test loader: {}\n'.format(len(train_loader),len(valid_loader),len(test_loader)))
    # train_loader 529   validation and test loader 175   
    model.train()
    for epoch in range(epochs):
        print('\n******[Epoch : {}]******'.format(epoch))
        flag = False
        for data,label in train_loader: 
            
            # data: [batch,mod,7]   label: [batch,mod,7]
            data,label = data.cuda(),label.cuda()
            out = model(data,label)
            loss = loss_MSE(out.reshape(out.shape[0],-1),label.reshape(label.shape[0],-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0)
            optimizer.step()
            if iters %20 == 0:
                print("iters: {}, loss: {}".format(iters,loss.item()))
            loss_dev = evaluate(model=model,data_loader=valid_loader)
            if loss_dev < best_loss:
                flag = True
                best_loss = loss_dev
                torch.save(model.state_dict(),'./best_parameters.pth')
                print('\niters: {}, loss: {}, dev_loss: {} '
                        .format(iters,loss.item(),loss_dev))
                test_loss = evaluate(model=model,data_loader=test_loader)
                print('\n test_loss: {} \n'.format(test_loss))
            model.train()
            iters += 1
        if not flag: # if no improve
            no_improve += 1
        if no_improve > 3:
            break
        

def evaluate(model,data_loader,mod=None,plt=None):
    model.eval()
    loss_total = 0
    loss_MSE = nn.L1Loss(reduce=True,size_average=True).cuda()
    with torch.no_grad():
        for data,label in data_loader:

            bs = data.shape[0]
            data,label = data.cuda(),label.cuda()
            out = model(data,label)
            loss = loss_MSE(out.reshape(bs,-1),label.reshape(bs,-1))
            loss_total += loss
            if plt:
                m_plot(data,label,out,mod)
            
    return loss_total / len(data_loader)


def test(model,data_loader,mod,plt=None):
    model.load_state_dict(torch.load('./best_parameters.pth'))
    loss = evaluate(model=model,data_loader=data_loader,mod=mod,plt=plt)
    return loss


def m_plot(data,label,pred,mod):
    pass
    

if __name__ == '__main__':
    random_state = 0   # 随机种子
    random.seed(random_state)
    np.random.seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.cuda.manual_seed(random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_state)
    torch.random.manual_seed(random_state)

    m_train()



