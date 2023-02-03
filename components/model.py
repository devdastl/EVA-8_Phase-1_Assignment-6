import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthwiseSeparable, self).__init__()
        self.depthwise = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel, bias=False)
        self.pointwise = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)

    def forward(self,x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
############################################################################################################################
class Netv1(nn.Module):
    def __init__(self, **kwargs):
        super(Netv1, self).__init__()
        if not (kwargs.get("in_channel") or kwargs.get("norm_type") or kwargs.get("drop_out")):
          raise TypeError("please specify in_channel, norm_type and drop_out value")
        
        self.in_channel = kwargs.get("in_channel")
        self.norm = "batch"
        self.dropout_value = kwargs.get("drop_out")

      # First block of CNN-------------- 30
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            self.norm_type(16,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #3x3

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #7x7

            DepthwiseSeparable(in_channel=32, out_channel=32),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #9x9


            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
        ) 

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #17x17

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #21x21

            DepthwiseSeparable(in_channel=64, out_channel=64),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #23x23


            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, bias=False, padding=1),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), 
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #32x32

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #36x36

            DepthwiseSeparable(in_channel=64, out_channel=64),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #38x38


            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=1, bias=False,padding=1),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), 
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #32x32

            DepthwiseSeparable(in_channel=32, out_channel=32),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            

            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), bias=False)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1x1x10 

    #function which setup normalization type based on string input arguments.
    def norm_type(self, feature, map_shape):
      norm_t = self.norm
      group = int(feature/2)
      if (norm_t == "batch"):
        return nn.BatchNorm2d(feature)
      elif (norm_t == "layer"):
        return nn.LayerNorm([feature, map_shape, map_shape], elementwise_affine=False)
      elif (norm_t == "group"):
        return nn.GroupNorm(num_groups=group, num_channels=feature)

      else:
        raise TypeError("please mention normalization technique from batchnorm, layernorm and groupnorm")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x=self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

########################################################################################################################################    
class Netv2(nn.Module):
    def __init__(self, **kwargs):
        super(Netv2, self).__init__()
        if not (kwargs.get("in_channel") or kwargs.get("norm_type") or kwargs.get("drop_out")):
          raise TypeError("please specify in_channel, norm_type and drop_out value")
        
        self.in_channel = kwargs.get("in_channel")
        self.norm = "batch"
        self.dropout_value = kwargs.get("drop_out")

      # First block of CNN-------------- 30
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            self.norm_type(16,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #3x3

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #7x7

            DepthwiseSeparable(in_channel=32, out_channel=32),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #9x9
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, stride=1, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=1, bias=False, dilation=2),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #15x15


        ) 

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #17x17

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #21x21

            DepthwiseSeparable(in_channel=64, out_channel=64),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #23x23
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=1, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), 

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=0, bias=False, dilation=2),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #29x29

            
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #32x32

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, dilation=2),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #36x36

            DepthwiseSeparable(in_channel=64, out_channel=64),
            self.norm_type(64,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #38x38
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), stride=1, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), 

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=1, bias=False, dilation=2),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #44x44


        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value), #32x32

            DepthwiseSeparable(in_channel=32, out_channel=32),
            self.norm_type(32,64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            

            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), bias=False)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # output_size = 1x1x10 

    #function which setup normalization type based on string input arguments.
    def norm_type(self, feature, map_shape):
      norm_t = self.norm
      group = int(feature/2)
      if (norm_t == "batch"):
        return nn.BatchNorm2d(feature)
      elif (norm_t == "layer"):
        return nn.LayerNorm([feature, map_shape, map_shape], elementwise_affine=False)
      elif (norm_t == "group"):
        return nn.GroupNorm(num_groups=group, num_channels=feature)

      else:
        raise TypeError("please mention normalization technique from batchnorm, layernorm and groupnorm")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x=self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
        #return x
    

