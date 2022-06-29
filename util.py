import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
np.random.seed(73)
import matplotlib.pyplot as plt
import time
import copy

from config import *

print()
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print()

def show(data,flag=True,title=""):
    plt.figure()
    plt.title(title)
    plt.imshow(data)
    if flag:
        plt.show()

def show3(im,out,label,title=""):
    fig, ax = plt.subplots(1,3)
    fig.suptitle(title, fontsize=20)
    ax[0].imshow(im)
    ax[0].set_title("Input")
    ax[1].imshow(out)
    ax[1].set_title("Output")
    ax[2].imshow(label)
    ax[2].set_title("Label")
    plt.show()


def data_gen(n=N,h=H,w=W,d=3,s=S):
    #dataset0 = []
    #data0 = np.random.rand(n//2,h,w,d)
    #for k in range(n//2):
    #    dataset0.append({"img" : torch.from_numpy(data0[k].transpose((2, 0, 1))), "label" : torch.from_numpy(np.zeros((h,w,1)).transpose((2, 0, 1)))})
    dataset1 = []
    data1 = np.random.rand(n,h,w,d)
    zeros = np.zeros((n,h,w,1))
    for k in range(n):
        indH = np.random.randint(0,h-s)
        indW = np.random.randint(0,w-s)
        for i in range(s):
            for j in range(s):
                try:
                    data1[k,indH+i,indW+j,0] = 1
                    data1[k,indH+i,indW+j,1] = 0
                    data1[k,indH+i,indW+j,2] = 0
                    zeros[k,indH+i,indW+j] = 1
                except:
                    continue
        dataset1.append({"img" : torch.from_numpy(data1[k]).permute(2,0,1), "label" : torch.from_numpy(zeros[k]).permute(2,0,1)})
    return dataset1

class ImData(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[int(idx)]

def get_eval_dataset(n=EVALS,h=H,w=W,d=3,s=S,batch_size=BS,flag=True):
    dataEval = ImData(data_gen(n,h,w,d,s))
    return {"eval" : torch.utils.data.DataLoader(dataEval,batch_size,shuffle=flag)}

def get_datasets(n=N,h=H,w=W,d=3,s=S,batch_size=BS,flag=True):
    dataTrain = ImData(data_gen(n,h,w,d,s))
    dataVal = ImData(data_gen(n//10,h,w,d,s))

    return {"train" : torch.utils.data.DataLoader(dataTrain,batch_size,shuffle=flag),
            "val"   : torch.utils.data.DataLoader(dataVal,batch_size,shuffle=flag)}

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

class UpSample(nn.Module):
    def __init__(self,nin,nout,kernel=5):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nin,nout,kernel,stride=1)
        self.bnn = nn.BatchNorm2d(nout)
    def forward(self,x):
        x = self.deconv(x)
        x = nn.functional.relu(x)
        x = self.bnn(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        print(x.shape)
        y = nn.ConvTranspose2d(512,128,1)(x)
        print(y.shape)
        return x

def rename_attribute(model, old_name, new_name):
    model._modules[new_name] = model._modules.pop(old_name)

def remove_attribute(model, name):
    model._modules.pop(name)

def initialize_model(feature_extract=True,classes=CLASSES):
    #resnet18
    model_ft = resnet18(weights=ResNet18_Weights.DEFAULT)
    set_parameter_requires_grad(model_ft, feature_extract)
    model = nn.Sequential(
        *list(model_ft.children())[:-2],
        nn.ReLU(),
        UpSample(512, 128, 8),
        UpSample(128, 32, 15),
        UpSample(32, 16, 29),
        UpSample(16, 8, 57),
        nn.ConvTranspose2d(8, classes, 113))
    input_size = 224
    return model, input_size

def set_update_params(model,v=False):
    params_to_update = model.parameters()
    if v: print("Params to learn:")
    feature_extract = True
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if v: print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                if v: print("\t",name)
    return params_to_update

class MaskLoss(nn.Module):
    def __init__(self,w=WLOSS):
        super(MaskLoss, self).__init__()
        self.w = w
 
    def forward(self, yPred, yTrue):        
        return self.w*torch.sum(torch.abs(yPred - yTrue))/yTrue.numel()

def train_model(model, dataloaders, criterion, optimizer, num_epochs=E):
    start = time.time()
    model = model.float()

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i_batch,batch in enumerate(dataloaders[phase]):
                inputs = batch["img"]
                labels = batch["label"]

                #if GPU equipped machine:
                #inputs = inputs.to(device)
                #labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Calculate loss
                    outputs = model(inputs.float())
                    loss = criterion(outputs, labels)

                    #_, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() #* inputs.size(0)
                if torch.all(torch.isclose(loss.float(),torch.Tensor(1))).item():
                    running_corrects += 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            elif phase == 'val' and (epoch_acc > best_acc or epoch_loss < best_loss):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_loss_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history

def evaluate(model,data,hist):
    model = model.float()
    model.eval()
    plt.plot(hist)
    plt.title("Loss")
    plt.show()
    for i_batch,batch in enumerate(data["eval"]):
        inputs = batch["img"]
        labels = batch["label"]

        outputs = model(inputs.float())

        show3(
            inputs.permute(0,2,3,1).numpy().reshape(224,224,3),
            outputs.permute(0,2,3,1).detach().numpy().reshape(224,224,1),
            labels.permute(0,2,3,1).numpy().reshape(224,224,1)
            )

        #show(inputs.permute(0,2,3,1).numpy().reshape(224,224,3),False,"Input")
        #show(outputs.permute(0,2,3,1).detach().numpy().reshape(224,224,1),False,"Output")
        #show(labels.permute(0,2,3,1).numpy().reshape(224,224,1),False,"Label")

        #plt.show()
