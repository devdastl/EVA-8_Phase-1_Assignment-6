#custom model to run test or evaluation loop 

import torch
import torch.nn as nn
import torch.nn.functional as F


class TestModel():
  def __init__(self, model, device, dataloader):
    self.model = model
    self.device = device
    self.dataloader = dataloader
    self.test_losses = []
    self.test_acc = []
    self.test_misc_img=[]
    self.test_misc_label=[]


  def test(self):
    self.model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in self.dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            for i in range(len(pred)): #loop through prediction and append wrong prediction one
              if pred[i] != target[i] and len(self.test_misc_img) != 10:
                self.test_misc_img.append(data[i])
                self.test_misc_label.append((pred[i].item(), target[i].item()))

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(self.dataloader.dataset)
    self.test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(self.dataloader.dataset),
        100. * correct / len(self.dataloader.dataset)))
    
    self.test_acc.append(100. * correct / len(self.dataloader.dataset))
