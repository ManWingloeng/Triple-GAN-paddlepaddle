import cifar10
from paddleops import *
import time

class triple_gan(object):
    def __init__(self, sess, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name, n, gan_lr, cla_lr, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.unlabelled_batch_size = unlabel_batch_size
        self.test_batch_size = 1000
        self.model_name = "TripleGAN"     # name for checkpoint
        if self.dataset_name == 'cifar10' :
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = z_dim
            self.y_dim = 10
            self.c_dim = 3

            self.learning_rate = gan_lr # 3e-4, 1e-3
            self.cla_learning_rate = cla_lr # 3e-3, 1e-2 ?
            self.GAN_beta1 = 0.5
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.alpha = 0.5
            self.alpha_cla_adv = 0.01
            self.init_alpha_p = 0.0 # 0.1, 0.03
            self.apply_alpha_p = 0.1
            self.apply_epoch = 200 # 200, 300
            self.decay_epoch = 50

            self.sample_num = 64
            self.visual_num = 100
            self.len_discrete_code = 10

            self.data_X, self.data_y, self.unlabelled_X, self.unlabelled_y, self.test_X, self.test_y = cifar10.prepare_data(n) # trainX, trainY, testX, testY

            self.num_batches = len(self.data_X) // self.batch_size

        else :
            raise NotImplementedError

    def D(self, x, y_, name='discriminator', is_test=False):
        with fluid.unique_name.guard(name+'_'):
            x = dropout(x, dropout_prob=0.2, is_test=False)
            y = reshape(y_, [-1, 1, 1, self.y_dim]) #ten classes
            x = conv_cond_concat(x, y)
            #weight norm?????????
            x = conv2d(x, num_filters=32, filter_size=[3,3], param_attr=wn('conv1'), name='conv1', act='lrelu')
            x = conv_cond_concat(x,y)
            x = conv2d(x, num_filters=32, filter_size=[3,3], stride=2, param_attr=wn('conv2'), name='conv2', act='lrelu')
            x = dropout(x, dropout_prob=0.2)

            x = conv2d(x, num_filters=64, filter_size=[3,3], param_attr=wn('conv3'), name='conv3', act='lrelu')
            x = conv2d(x, num_filters=64, filter_size=[3,3], stride=2, param_attr=wn('conv4'), name='conv4', act='lrelu')
            x = dropout(x, dropout_prob=0.2)

            x = conv2d(x, num_filters=128, filter_size=[3,3], param_attr=wn('conv5'), name='conv5', act='lrelu')
            x = conv2d(x, num_filters=128, filter_size=[3,3], param_attr=wn('conv6'), name='conv6', act='lrelu')

            x = Global_Average_Pooling(x)
            #IcGAN 每一层都要concat一下

