import argparse
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import math
import facades_loader as fl
import os
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=200,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='the batch size')
    parser.add_argument('--learning-rate', type=float, default = 0.0002,
                        help='learning rate')
    parser.add_argument('--eval', action='store_true',
                        help='eval mode')
    parser.add_argument('--save', action='store_true',
                        help='save on')
    parser.add_argument('--folder', type= str,default ='facades',
                         help='dataset folder name')
    return parser.parse_args()


'''

global variable for simplicity

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
is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)




'''
m  =model

'''


def drawlossplot(epoch, loss_g, loss_d, e):
    g_x = np.linspace(0, len(loss_g), len(loss_g))
    f, ax = plt.subplots(1)

    plt.plot(g_x, loss_g, label='loss_g')
    plt.plot(g_x, loss_d, label='loss_d')
    ax.set_xlim(0, epoch)

    plt.title('Generative Adversarial Network Loss Graph')
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.legend()
    plt.savefig("pix2pix_loss_epoch%d" % e)
    plt.close()


def batch_norm_wrapper(inputs, i,decay = 0.999):
    in_sh = inputs.get_shape().as_list()
    epsilon = 0.0001
    scale = tf.get_variable("scale"+str(i),in_sh, initializer = tf.ones_initializer)
    beta = tf.get_variable("beta"+str(i),in_sh, initializer = tf.zeros_initializer)
    pop_mean = tf.get_variable("pop_mean"+str(i),in_sh, initializer = tf.zeros_initializer,trainable = False)
    pop_var = tf.get_variable("pop_var"+str(i),in_sh, initializer = tf.ones_initializer,trainable = False)


    def bn_training():
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
         return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var,beta,scale, epsilon)
    def bn_evaluation():
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)
    return tf.cond(is_training, bn_training , bn_evaluation)

def convblocklayer(x, out_ch,kernel_size,stride,i):
    x_s = x.get_shape().as_list()
    #feature = tf.Variable(tf.random_normal([kernel_size,kernel_size,x_s[3],out_ch], stddev = 0.01))
    feature = tf.get_variable("feature"+str(i),[kernel_size,kernel_size,x_s[3],out_ch])
    g = tf.nn.conv2d(x,feature,[1,stride,stride,1],'SAME')
    g = batch_norm_wrapper(g,i)
    g = tf.nn.leaky_relu(g)
    print(g)
    return g

def deconvblocklayer(x,out_ch,kernel_size,stride,padding,i,dropout = True):
    x_s = x.get_shape().as_list()
    #feature = tf.Variable(tf.random_normal([kernel_size,kernel_size,out_ch,x_s[3]], stddev = 0.01))
    feature = tf.get_variable("feature"+str(i),[kernel_size,kernel_size,out_ch,x_s[3]])
    o_s = stride*(x_s[2]-1) + kernel_size -2*padding 
    g = tf.nn.conv2d_transpose(x,feature,[x_s[0],o_s,o_s,out_ch],[1,stride,stride,1])
    g = batch_norm_wrapper(g,i)
    if dropout:
        g = tf.nn.dropout(g,keep_prob)
    g = tf.nn.relu(g)
    print(g)
    return g

def generator(input,output, ini_ch_size):
     #g = tf.concat([input,label_in],3)
     g1 = convblocklayer(input, ini_ch_size,4,2, 0)
     g2 = convblocklayer(g1, 2*ini_ch_size, 4, 2, 1)
     g3 = convblocklayer(g2, 4*ini_ch_size, 4, 2, 2)
     g4 = convblocklayer(g3, 8*ini_ch_size, 4, 2, 3)
     g5 = convblocklayer(g4, 8*ini_ch_size, 4, 2, 4)
     g6 = convblocklayer(g5, 8 * ini_ch_size, 4, 2, 5)
     g7 = convblocklayer(g6, 8 * ini_ch_size, 4, 2, 6)
     feature = tf.get_variable("feature_c", [4, 4, 8*ini_ch_size, 8*ini_ch_size])
     g8 = tf.nn.relu(tf.nn.conv2d(g7, feature, [1, 2, 2, 1], 'SAME'))
     g9 = deconvblocklayer(g8,8*ini_ch_size,4,2,1,7)
     g10 = deconvblocklayer(tf.concat([g9, g7], 3),8*ini_ch_size,4,2,1,8)
     g11 = deconvblocklayer(tf.concat([g10, g6], 3), 8*ini_ch_size, 4, 2, 1, 9)
     g12 = deconvblocklayer(tf.concat([g11, g5], 3), 8*ini_ch_size, 4, 2, 1, 10,False)
     g13 = deconvblocklayer(tf.concat([g12, g4], 3), 4*ini_ch_size, 4, 2, 1, 11,False)
     g14 = deconvblocklayer(tf.concat([g13, g3], 3), 2*ini_ch_size, 4, 2, 1, 12, False)
     g15 = deconvblocklayer(tf.concat([g14, g2], 3), 1*ini_ch_size, 4, 2, 1, 13, False)
     o_c = output.get_shape().as_list()
     #feature = tf.Variable(tf.random_normal([4,4,o_c[3],128], stddev = 0.01))
     feature = tf.get_variable("feature",[4,4,o_c[3],1*ini_ch_size])
     g_s = g15.get_shape().as_list()
     o_s = 2*(g_s[2]-1) + 4 -2*1
     g = tf.nn.conv2d_transpose(g15,feature,[g_s[0], o_s, o_s, o_c[3] ],[1,2,2,1])
     g = tf.nn.sigmoid(g)
     print(g)
     return g

def discriminator(input,label_ch, ini_ch_size):
     d1 = convblocklayer(input,ini_ch_size,4,2,0)
     d2 = convblocklayer(label_ch,ini_ch_size,4,2,1)
     d3 = tf.concat([d1,d2],3)
     d = convblocklayer(d3,ini_ch_size*2,4,2,2)
     d = convblocklayer(d,ini_ch_size*4,4,2,3)
     d = convblocklayer(d,ini_ch_size*8,4,1,4)
     feature = tf.get_variable("feature",[4,4,d.shape[3],1])
     d = tf.nn.conv2d(d,feature,[1,1,1,1],'SAME')
     d = tf.nn.sigmoid(d)
     print(d)
     return d

class GAN(object):
      def __init__(self,params,input,output,ini_ch_size):
          self.bn = params.batch_size
          self.lr = params.learning_rate
          self.epoch = params.num_steps
          with tf.variable_scope('g'):
           self.g = generator(input,output,ini_ch_size)
          with tf.variable_scope('d', reuse=tf.AUTO_REUSE):
           self.d_r = discriminator(output,input,ini_ch_size)
           self.d_f = discriminator(self.g,input,ini_ch_size)
          self.in_ph = input
          self.ou_ph = output
          vars = tf.trainable_variables()
          self.d_params = [v for v in vars if v.name.startswith('d/')]
          self.g_params = [v for v in vars if v.name.startswith('g/')]
          self.loss_d = tf.reduce_mean(-tf.log(self.d_r) - tf.log(1 - self.d_f))
          self.loss_g = tf.reduce_mean(-tf.log(self.d_f))+tf.reduce_mean(tf.abs(self.g - self.ou_ph))
          self.optm_d = tf.train.AdamOptimizer(self.lr).minimize(self.loss_d, var_list=self.d_params)
          self.optm_g = tf.train.AdamOptimizer(self.lr).minimize(self.loss_g, var_list=self.g_params)
      def save(self,sess,i):
          saver = tf.train.Saver()
          saver.save(sess,'./checkpoints'+str(i)+'.ckpt', write_meta_graph=False)
      def load(self,sess,i):
          saver = tf.train.Saver()
          saver.restore(sess, './checkpoints'+str(i)+'.ckpt')




def train(sess,model, trl, i):
    # saver = tf.train_Saver()
    print("epoch :%s" % i)
    real_batch_img, fake_batch_img = trl.getbn(True)
    e_loss_g = 0
    e_loss_d = 0
    while real_batch_img != []:
        loss_value_d, _ = sess.run((model.loss_d, model.optm_d), feed_dict={model.in_ph: fake_batch_img, model.ou_ph: real_batch_img, is_training: True, keep_prob :0.5})
        loss_value_g, _ = sess.run((model.loss_g, model.optm_g), feed_dict={model.in_ph: fake_batch_img, model.ou_ph: real_batch_img, is_training: True, keep_prob :0.5})
        e_loss_g += loss_value_d
        e_loss_d += loss_value_g

        real_batch_img, fake_batch_img = trl.getbn(True)


    return e_loss_g / trl.lens, e_loss_d / trl.lens


def eval(sess,model, trl, i):


    print("eval epoch :%s" % i)
    path = "result"+str(i)
    if not os.path.exists(path):
     os.mkdir(path)
    real_batch_img, fake_batch_img = trl.getbn(True,False)
    j = 0
    while real_batch_img != []:
        g = sess.run(model.g,  feed_dict={model.in_ph: fake_batch_img, is_training: True, keep_prob :0.5})
        g = np.concatenate((g,fake_batch_img,real_batch_img),2)
        g *= 255
        for t in range(g.shape[0]):
            img = Image.fromarray(g[t].astype('uint8'))
            img.save(path+'/eval'+str(j)+'.jpg','JPEG' )
            j += 1
        real_batch_img, fake_batch_img = trl.getbn(True,False)
        #real_batch_img = []


def main(args):
    train_dataset = fl.imgldr(fl.mypath, args.batch_size)
    eval_dataset = fl.imgldr(fl.mypath.replace('train', 'val'), args.batch_size)

    input = tf.placeholder(tf.float32,  [args.batch_size, y_size,x_size,in_ch])
    output = tf.placeholder(tf.float32, [args.batch_size,y_size,x_size,out_ch] )
    gan = GAN(args,input,output,ini_ch_size)
    a_loss_g = []
    a_loss_d = []

    with tf.Session() as sess:
     init = tf.global_variables_initializer()
     sess.run(init)

     for i in range(args.num_steps):
      e_loss_g, e_loss_d = train(sess,gan, train_dataset,i)
      a_loss_g.append(e_loss_g)
      a_loss_d.append(e_loss_d)
      print(e_loss_d)
      drawlossplot(args.num_steps, a_loss_g, a_loss_d, i)
      if i % 10 == 0:
       eval(sess,gan, eval_dataset,i)
       if args.save:
        gan.save(sess,i)

#    writer = tf.summary.FileWriter('.')
#    writer.add_graph(tf.get_default_graph())


if __name__ == '__main__' :
    main(parse_args())

