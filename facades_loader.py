import os
import torch
from PIL import Image
from PIL import ImageOps
import numpy as np


mypath = '/home/jusunglee/vision_db/edges2shoes/train/'

resize_img = 256

class imgldr(object):
    def __init__(self,path,batch_size):
         self.path = path
         self.bn   = batch_size
         self.cnt  = 0
         self.getlist()

    def getlist(self):
         self.list = [f for f in os.listdir(self.path)]
         self.lens = len(self.list)
         print("len : %s" %self.lens)
    def getbn(self,resize = False,flip = True):
         if self.cnt+self.bn > self.lens:
          self.cnt = 0
          return [], []
         else :
          #print(self.cnt)
          real_batch_img = []
          fake_batch_img = []
          info_im = Image.open(self.path + self.list[0])
          info_im = np.array(info_im)


          ch = info_im.shape[-1]
          for i in range(self.bn):
           file = self.path + self.list[i+self.cnt]
           im = Image.open(file)
           if resize :
            im=im.resize((2*resize_img+12,resize_img+12))
           offset_x = np.random.randint(0,5)
           offset_y = np.random.randint(0,5)
           real = im.crop((offset_x,offset_y,offset_x+resize_img,offset_y+resize_img))
           fake = im.crop((offset_x+resize_img+6,offset_y,offset_x+2*resize_img+6,offset_y+resize_img))

           if offset_x >2 and flip:
               real = ImageOps.mirror(real)
               fake = ImageOps.mirror(fake)
           real = np.moveaxis((np.array(real)/255.0).astype(np.float32),-1,0)
           fake = np.moveaxis((np.array(fake)/255.0).astype(np.float32), -1,0)
           real_batch_img.append(real)
           fake_batch_img.append(fake)

          real_batch_img = np.array(real_batch_img).reshape(-1,ch,resize_img,resize_img)
          fake_batch_img = np.array(fake_batch_img).reshape(-1, ch, resize_img, resize_img)
          '''
          print(real_batch_img[0].shape)
          temp = np.moveaxis(real_batch_img[0],0,-1)
          print(temp.shape)
          de_im = Image.fromarray(temp)
          de_im.show()
          '''
          self.cnt += self.bn
          return real_batch_img, fake_batch_img








    #cel = imgldr(mypath,1)
#cel.getbn()
#cel.getbn()
     
