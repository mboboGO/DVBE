import torch
import torch.nn as nn

PRIMITIVES = [
    'fc_relu',
    'skip_connect',
    'gcn',
    'none',
]

class MixedOp(nn.Module):

    def __init__(self, c_in, c_out,adj=None):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        PRIMITIVES = PRIMITIVES

        if c_in!=c_out:
            self.reduction = True
            self.preprocess = nn.Sequential(
                nn.Linear(c_in,c_out),
                nn.LeakyReLU(),
            )
        else:
            self.reduction = False
			
        for primitive in PRIMITIVES:
            if primitive=='gcn':
                op = OPS[primitive](c_in,c_out,adj)
            else:
                op = OPS[primitive](c_in,c_out)
            self._ops.append(op)
            
    def forward(self, x, weights):
        if self.reduction:
            x = self.preprocess(x)
        
        #for w, op in zip(weights, self._ops):
        #    print(op(x).size())
        
        return sum(w * op(x) for w, op in zip(weights, self._ops))
        
        
OPS = {
  'none' : lambda C1,C2: Zero(C1,C2),
  'fc_relu' : lambda C1,C2: FC_RELU(C1,C2),
  'skip_connect' : lambda C1,C2: Identity_fc(C1,C2),
  'skip_connect_conv' : lambda C1,C2: Identity_conv(C1,C2),
  'c_att' : lambda C1,C2: C_ATT(C1, C2),
  'gcn' : lambda C1,C2,adj: GraphConv(C1,C2,adj),
  'avg_pool_3x3' : lambda C1,C2: nn.AvgPool2d(3, stride=1, padding=1),
  'max_pool_3x3' : lambda C1,C2: MaxPool(C1,C2, 3,1,1),
  'sep_conv_3x3' : lambda C1,C2: SepConv(C1, C2, 3, 1, 1),
  #'sep_conv_5x5' : lambda Ce: SepConv(C, C, 5, 1, 2, affine=affine),
  #'sep_conv_7x7' : lambda C: SepConv(C, C, 7, 1, 3, affine=affine),
  'dil_conv_3x3' : lambda C1,C2: DilConv(C1, C2, 3, 1, 2, 2),
  #'dil_conv_5x5' : lambda C: DilConv(C, C, 5, 1, 4, 2, affine=affine),
  'conv_3x3' : lambda C1,C2: Conv(C1, C2, 3, 1, 1),
}


class DilConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
      nn.ReLU(inplace=False),
      )

  def forward(self, x):
    return self.op(x)

class Conv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super(Conv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
      nn.ReLU(inplace=False),
      )

  def forward(self, x):
    return self.op(x)

class MaxPool(nn.Module):
  def __init__(self, c_in, c_out, kernel_size, stride, padding):
    super(MaxPool, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
      nn.ReLU(inplace=False),
      )
    self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
    if c_in!=c_out:
        self.flag = True
    else:
        self.flag = False 

  def forward(self, x):
    if self.flag:
        x = self.conv(x)
    return self.maxpool(x)
     
class SepConv(nn.Module):
  def __init__(self, C_in, C_out, kernel_size, stride, padding):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.ReLU(inplace=False),
      )

  def forward(self, x):
    return self.op(x)
    
class FC_RELU(nn.Module):

  def __init__(self, C_in, C_out):
    super(FC_RELU, self).__init__()
    self.op = nn.Sequential(
        nn.Linear(C_in,C_out),
        nn.LeakyReLU(),
    )

  def forward(self, x):
    return self.op(x)

class Identity_fc(nn.Module):

  def __init__(self,c_in,c_out):
    super(Identity_fc, self).__init__()
    self.c_in = c_in
    self.c_out = c_out
    self.op = nn.Sequential(
        nn.Linear(c_in,c_out),
        nn.LeakyReLU(),
    )

  def forward(self, x):
    if self.c_in == self.c_out:
        return x
    else:
        return(self.op(x))
        
class Identity_conv(nn.Module):

  def __init__(self,c_in,c_out):
    super(Identity_conv, self).__init__()
    self.c_in = c_in
    self.c_out = c_out
    self.op = nn.Sequential(
      nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
      nn.ReLU(inplace=False),
      )

  def forward(self, x):
    if self.c_in == self.c_out:
        return x
    else:
        return(self.op(x))

class Zero(nn.Module):

  def __init__(self,c_in,c_out):
    super(Zero, self).__init__()
    self.c_out = c_out

  def forward(self, x):
    if len(x.size())>2:
      return torch.zeros(x.size(0),self.c_out,x.size(2),x.size(3)).cuda()
    else:
      return torch.zeros(x.size(0),self.c_out).cuda()


class C_ATT(nn.Module):

  def __init__(self, C_in, C_out):
    super(C_ATT, self).__init__()
    assert C_out % 2 == 0
    self.op = nn.Sequential(
        nn.Linear(C_in,C_in/16),
        nn.Linear(C_in/16,C_out),
        nn.Sigmoid(),
    )

  def forward(self, x):
    out = x+x*self.op(x)
    return out

class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, adj, dropout=False, relu=True):
        super(GraphConv,self).__init__()

        self.adj = adj

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
            
        self.fc = nn.Linear(in_channels,out_channels)
        if relu:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = None

    def forward(self, inputs):
        if self.dropout is not None:
            inputs = self.dropout(inputs)
            
        outputs = self.fc(inputs)
        outputs = torch.mm(self.adj,outputs)

        if self.relu is not None:
            outputs = self.relu(outputs)
            
        outputs = outputs# + inputs
        return outputs
