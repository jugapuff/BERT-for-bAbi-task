from dataloader import bAbi_Dataset
import torch
import torch.nn as nn
from model import model
from pytorch_transformers import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU:" + str(torch.cuda.get_device_name(0)))
    
my_model = model()
my_model.to(device)

optimizer = AdamW(my_model.parameters())
criterion = nn.NLLLoss()


EPOCHS = 10
for epoch in range(1, EPOCHS+1):
    
    my_model.train()
    
    train_loss = 0
    length = 0
    for tokens_tensor, segments_tensors, att_mask, pos_id, trg in data_loader:
        output = my_model(tokens_tensor.to(device), segments_tensors.to(device), att_mask.to(device), pos_id.to(device))
        loss = criterion(output, trg.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        length+=1
        train_loss += loss.item()
        if length % 10 == 0:
          print("\t\t{:3}/25000 : {}".format(length, train_loss / length))
        
    epoch_loss = train_loss / length
    print("##################")
    print("{} epoch Loss : {:.4f}".format(epoch, epoch_loss))
  