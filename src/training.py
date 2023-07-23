from torch.optim import Adam
from torch import nn
from tqdm import tqdm
import torch
from dataloader import load_bert_classifer_data

def train(model, train_json_file_list, valid_json_file_list, learning_rate, epochs):
    train_dataloader = load_bert_classifer_data(train_json_file_list, batch_size=4, shuffle=True)
    val_dataloader = load_bert_classifer_data(valid_json_file_list, batch_size=2, shuffle=True)
    
  # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
      # 定义两个变量，用于存储训练集的准确率和损失
            total_acc_train = 0
            total_loss_train = 0
      # 进度条函数tqdm
            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
                # 模型计算
                output = model(input_id, mask)
                # 计算损失
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                # 计算精度
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
            # 模型更新
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_val = 0
            total_loss_val = 0
            # 不需要计算梯度
            with torch.no_grad():
                # 循环获取数据集，并用训练好的模型进行验证
                for val_input, val_label in val_dataloader:
                    # 如果有GPU，则使用GPU，接下来的操作同训练
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / (15600*2): .3f} 
              | Train Accuracy: {total_acc_train / (15600*2): .3f} 
              | Val Loss: {total_loss_val / 15600: .3f} 
              | Val Accuracy: {total_acc_val / 15600: .3f}''') 
            
if __name__ == "__main__":
    from classifer import BertClassifier
    model_path = '/home/data/zhengyuhu/bert/bert-base-uncased'
    model = BertClassifier(model_path=model_path)
    train_json_file_list = ['../data/train.json']
    valid_json_file_list = ['../data/valid.json']
    learning_rate = 1e-6
    epochs = 5

    train(model, train_json_file_list, valid_json_file_list, learning_rate, epochs)