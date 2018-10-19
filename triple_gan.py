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

    def D(self, x, y_, name='Discriminator', is_test=False):
        with fluid.unique_name.guard(name+'_'):
            x = dropout(x, dropout_prob=0.2, is_test=False)
            y = reshape(y_, [-1, 1, 1, self.y_dim]) #ten classes
            x = conv_cond_concat(x, y)
            #weight norm in paddlepaddle has finished
            x = conv2d(x, num_filters=32, filter_size=[3,3], param_attr=wn('conv1'), name='conv1', act='lrelu')
            x = conv_cond_concat(x, y)
            x = conv2d(x, num_filters=32, filter_size=[3,3], stride=2, param_attr=wn('conv2'), name='conv2', act='lrelu')
            x = dropout(x, dropout_prob=0.2)
            x = conv_cond_concat(x, y)

            x = conv2d(x, num_filters=64, filter_size=[3,3], param_attr=wn('conv3'), name='conv3', act='lrelu')
            x = conv_cond_concat(x, y)
            x = conv2d(x, num_filters=64, filter_size=[3,3], stride=2, param_attr=wn('conv4'), name='conv4', act='lrelu')
            x = dropout(x, dropout_prob=0.2)
            x = conv_cond_concat(x, y)

            x = conv2d(x, num_filters=128, filter_size=[3,3], param_attr=wn('conv5'), name='conv5', act='lrelu')
            x = conv_cond_concat(x, y)
            x = conv2d(x, num_filters=128, filter_size=[3,3], param_attr=wn('conv6'), name='conv6', act='lrelu')
            x = conv_cond_concat(x, y)
            
            x = Global_Average_Pooling(x)
            x = flatten(x)
            x = concat(x, y_)
            #IcGAN 每一层都要concat一下

            # MLP??
            x_logit = fc(x, 1, name='fc')
            out = sigmod(x_logit, name='sigmod')

            return out, x_logit, x


    def G(self, z, y, name='Generator', is_test=False):
        with fluid.unique_name.guard(name+'_'):
            zy=concat(z,y)
            zy = fc(zy, 8192, name='fc', act='relu')
            zy = bn(zy, name='bn', act='relu')
            zy = reshape(zy, [-1, 4, 4, 512])
            y = reshape(y, [-1, 1, 1, self.y_dim])
            zy = conv_cond_concat(zy, y)
            zy = deconv(zy, num_filters=256, filter_size=[5, 5], stride=2, name='deconv1')
            zy = bn(zy, act='relu')

            zy = conv_cond_concat(zy, y)
            zy = deconv(zy, num_filters=128, filter_size=[5, 5], stride= 2, name='deconv2')
            zy = bn(zy, act='relu')

            zy = conv_cond_concat(zy, y)
            zy = deconv(zy, num_filters=3, filter_size=[5, 5], stride=2, 
                            param_attr=wn(name='deconv3'), name='deconv3', act='tanh')
            
            return zy

    def C(self, x, name='Classifier', is_test=False):
        x = gaussian_noise_layer(x, std=0.15)
        x = conv2d(x, num_filters=128, filter_size=[3,3], act='lrelu', param_attr=wn('conv1'), name='conv1')
        x = conv2d(x, num_filters=128, filter_size=[3,3], act='lrelu', param_attr=wn('conv2'), name='conv2')
        x = conv2d(x, num_filters=128, filter_size=[3,3], act='lrelu', param_attr=wn('conv3'), name='conv3')
        x = max_pooling(x , pool_size=[2,2])
        x = dropout(x, dropout_prob=0.5)

        x = conv2d(x, num_filters=256, filter_size=[3,3], act='lrelu', param_attr=wn('conv4'), name='conv4')
        x = conv2d(x, num_filters=256, filter_size=[3,3], act='lrelu', param_attr=wn('conv5'), name='conv5')
        x = conv2d(x, num_filters=256, filter_size=[3,3], act='lrelu', param_attr=wn('conv5'), name='conv5')
        x = max_pooling(x , pool_size=[2,2])
        x = dropout(x, dropout_prob=0.5)

        x = conv2d(x, num_filters=512, filter_size=[3,3], act='lrelu', param_attr=wn('conv5'), name='conv5')
        x = nin(x, 256)




