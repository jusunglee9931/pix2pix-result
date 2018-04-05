import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import math
import edges_loader as fl
import os
from PIL import Image


#seed = 11
#np.random.seed(seed)
#torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=300,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='the batch size')
    parser.add_argument('--learning-rate', type=float, default = 0.0002,
                        help='learning rate')
    parser.add_argument('--eval', action='store_true',
                        help='eval mode')
    parser.add_argument('--save', action='store_true',
                        help='save on')
    return parser.parse_args()


'''
variables
'''
x_size =256
y_size =256
in_ch = 3
out_ch =3
z_size =100
label_size = 10
batch_size = parse_args().batch_size
ini_ch_size = 64
outputsize = 30



'''
m  =model

'''
def drawlossplot( epoch,loss_g,loss_d,e):
    g_x = np.linspace(0, len(loss_g), len(loss_g))
    f, ax = plt.subplots(1)
   
    plt.plot(g_x, loss_g, label='loss_g')
    plt.plot(g_x, loss_d, label='loss_d')
    ax.set_xlim(0, epoch)

    plt.title('Generative Adversarial Network Loss Graph')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.legend()
    plt.savefig("cifar_cdcgan_loss_epoch%d" %e)
    plt.close()


def weights_init(self):
    classname = self.__class__.__name__
    if classname.find('Conv') != -1:
        self.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        self.weight.data.normal_(1.0, 0.02)
        self.bias.data.fill_(0)

def convblocklayer(in_ch,out_ch,stride = 2):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size =4, stride = stride,padding = 1),
                         nn.BatchNorm2d(out_ch),
                         nn.LeakyReLU(0.2)
                         )
def deconvblocklayer(in_ch,out_ch,pad,dropout = True):
    if dropout:
     return nn.Sequential(nn.ConvTranspose2d(in_ch,out_ch,kernel_size = 4, stride = 2,padding = pad),
                         nn.BatchNorm2d(out_ch),
                          nn.Dropout(0.5),
                         nn.ReLU()
                         )
    else:
     return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=pad),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU()
                             )


class generator(nn.Module):
      def __init__(self,img_channel,o_channel,ini_ch_size):
          super(generator, self).__init__()
          self.layer1 = convblocklayer(img_channel,ini_ch_size)
          self.layer2 = convblocklayer(ini_ch_size,2*ini_ch_size)
          self.layer3 = convblocklayer(2*ini_ch_size,4*ini_ch_size)
          self.layer4 = convblocklayer(4*ini_ch_size,8*ini_ch_size)
          self.layer5 = convblocklayer(8*ini_ch_size,8*ini_ch_size)
          self.layer6 = nn.Conv2d(8*ini_ch_size,8*ini_ch_size,kernel_size =4, stride = 2,padding = 1)
          self.layer7 = deconvblocklayer(2*8*ini_ch_size,8*ini_ch_size,1)
          self.layer7_no_drop = deconvblocklayer(2 * 8 * ini_ch_size, 8 * ini_ch_size, 1,False)
          self.layer8 = deconvblocklayer(2*8*ini_ch_size,4*ini_ch_size,1,False)
          self.layer9 = deconvblocklayer(2*4*ini_ch_size,2*ini_ch_size,1,False)
          self.layer10= deconvblocklayer(2*2*ini_ch_size,ini_ch_size,1,False)
          self.layer11 = deconvblocklayer( 8 * ini_ch_size, 8 * ini_ch_size, 1)
          self.deconv = nn.ConvTranspose2d(2*ini_ch_size,o_channel, kernel_size =4,stride = 2, padding =1)
          self.activ  = nn.Sigmoid()
          self.relu   = nn.ReLU()
      def forward(self, x):
          out1 = self.layer1(x)
          out2 = self.layer2(out1)
          out3 = self.layer3(out2)
          out4 = self.layer4(out3)
          out5 = self.layer5(out4)
          out6 = self.layer5(out5)
          out7 = self.layer5(out6)
          out8 = self.relu(self.layer6(out7))
          out9 = self.layer11(out8)
          out10 = self.layer7(torch.cat((out9,out7),1))
          out11 = self.layer7(torch.cat((out10,out6),1))
          out12 = self.layer7_no_drop(torch.cat((out11,out5),1))
          out13 = self.layer8(torch.cat((out12, out4), 1))
          out14 = self.layer9(torch.cat((out13, out3), 1))
          out15 = self.layer10(torch.cat((out14, out2), 1))
          out16 = self.deconv(torch.cat((out15, out1), 1))
          out = self.activ(out16)
          return out


class discriminator(nn.Module):
      def __init__(self, img_channel,ini_ch_size):
          super(discriminator,self).__init__()
          self.layer1 =convblocklayer(img_channel,int(ini_ch_size/2))
          self.layer2 =convblocklayer(ini_ch_size,2*ini_ch_size)
          self.layer3 =convblocklayer(2*ini_ch_size,4*ini_ch_size)
          self.layer4 =convblocklayer(4 * ini_ch_size, 8 * ini_ch_size,1)
          self.conv5 = nn.Conv2d(8*ini_ch_size,1, kernel_size =4,stride = 1, padding = 1)
          self.activ = nn.Sigmoid()
      def forward(self,x,y):
          out1 = self.layer1(x)
          out2 = self.layer1(y)
          out = self.layer2(torch.cat((out1,out2),1))
          out = self.layer3(out)
          out = self.layer4(out)
          out = self.conv5(out)
          out = self.activ(out)
          return out




class GAN(object):
      def __init__(self,params,in_ch,o_ch,ini_ch_size):
          self.g = generator(in_ch,o_ch,ini_ch_size)
          self.d = discriminator(in_ch,ini_ch_size)
          self.g.apply(weights_init)
          self.d.apply(weights_init)
          self.g.cuda(0)
          self.d.cuda(0)
          self.batch_size = params.batch_size
          self.lr = params.learning_rate
          self.ct = nn.BCELoss()
          self.l1loss =nn.L1Loss()
          self.lambda_c = 100
          self.g_opt = torch.optim.Adam(self.g.parameters(),lr = self.lr, betas = (0.5,0.999))
          self.d_opt = torch.optim.Adam(self.d.parameters(),lr = self.lr, betas = (0.5,0.999))
          self.epoch = params.num_steps
      def save(self,i):
          torch.save(self.g,"g_epoch "+str(i)+".pt")
          torch.save(self.d,"d_epoch "+str(i)+".pt")
      def load(self,i):
          self.g = torch.load("g_epoch "+str(i)+".pt")
          self.d = torch.load("d_epoch "+str(i)+".pt")

def train(model,trl,i):
    
    ones = Variable(torch.ones(model.batch_size,1,outputsize,outputsize).cuda())
    zeros = Variable(torch.zeros(model.batch_size,1,outputsize,outputsize).cuda())


     
    print("epoch :%s" %i)
    real_batch_img, fake_batch_img = trl.getbn(True)
    e_loss_g = 0
    e_loss_d = 0
    while real_batch_img != []:
        ds = Variable(torch.from_numpy(real_batch_img).cuda())
        gs = Variable(torch.from_numpy(fake_batch_img).cuda())

        model.d_opt.zero_grad()

        d1 = model.d(ds,gs)
        g = model.g(gs)
        d2 = model.d(g,gs)
      
        loss_d1 = model.ct(d1,ones)
        loss_d2 = model.ct(d2,zeros)

        loss = loss_d1 + loss_d2
        loss.backward(retain_graph=True)
        model.d_opt.step()


        model.g_opt.zero_grad()
        
        loss_g = model.ct(d2,ones)+model.lambda_c*model.l1loss(g,ds)
        loss_g.backward()
        model.g_opt.step()

        real_batch_img, fake_batch_img = trl.getbn(True)
        e_loss_g += loss_g.data[0]
        e_loss_d += loss.data[0]
        #real_batch_img = []

    return e_loss_g / trl.lens , e_loss_d / trl.lens

def eval(model, trl, i):


    print("eval epoch :%s" % i)
    path = "result"+str(i)
    if not os.path.exists(path):
     os.mkdir(path)
    real_batch_img, fake_batch_img = trl.getbn(True,False)
    j = 0
    while real_batch_img != []:
        gs = Variable(torch.from_numpy(fake_batch_img).cuda())
        g = model.g(gs)
        g = g.data.cpu().numpy()
        #g = g.reshape(-1, out_ch, y_size, x_size)
        g = np.concatenate((g,fake_batch_img,real_batch_img),3)
        for t in range(g.shape[0]):
            tp = np.moveaxis(g[t],0,-1)
            tp *= 255
            img = Image.fromarray(tp.astype('uint8'))
            img.save(path+'/eval'+str(j)+'.jpg','JPEG' )
            j += 1
        real_batch_img, fake_batch_img = trl.getbn(True,False)
        #real_batch_img = []

  






def main(args):
   
    model = GAN(args,in_ch,out_ch,ini_ch_size)
    train_dataset = fl.imgldr(fl.mypath, args.batch_size)
    eval_dataset = fl.imgldr(fl.mypath.replace('train','val'), args.batch_size)
    a_loss_g = []
    a_loss_d = []



    for i in range(args.num_steps):
     e_loss_g, e_loss_d = train(model, train_dataset,i)
     a_loss_g.append(e_loss_g)
     a_loss_d.append(e_loss_d)
     print(e_loss_d)
     drawlossplot(args.num_steps, a_loss_g, a_loss_d, i)
     if i % 10 == 0:
       eval(model, eval_dataset,i)
       if args.save:
         model.save(i)


if __name__ == '__main__':
    main(parse_args())










