import cifar10
from paddleops import *
import time
import paddle.fluid as fluid
import paddle


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
            self.alpha_p = 0.1 ##??

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
            out = sigmoid(x_logit, name='sigmoid')

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
        x = nin(x, 256, param_attr=wn('nin1'), act='lrelu')
        x = nin(x, 128, param_attr=wn('nin1'), act='lrelu')
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = fc(x, 10, param_attr=wn('fc1'), name=name+'_fc1')
        out = softmax(x)

        return x, out

    def loss(self, x, label):
        return fluid.layers.mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(
                x=x, label=label))

    def train(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        unlabel_bs = self.unlabelled_batch_size
        test_bs = self.test_batch_size
        alpha = self.alpha
        alpha_cla_adv = self.alpha_cla_adv

        # images
        self.inputs = fluid.layers.data(shape=[bs] + image_dims, name='real_images')
        self.unlabelled_inputs = fluid.layers.data(shape=[unlabel_bs] + image_dims, name='unlabelled_images')
        self.test_inputs = fluid.layers.data(shape=[test_bs] + image_dims, name='test_images')

        # labels
        self.y = fluid.layers.data(shape=[bs, self.y_dim], name='y')
        self.unlabelled_inputs_y = fluid.layers.data(shape=[unlabel_bs, self.y_dim], name='unlabelled_images_y')
        self.test_label = fluid.layers.data(shape=[test_bs, self.y_dim], name='test_label')
        self.visual_y = fluid.layers.data(shape=[self.visual_num, self.y_dim], name='visual_y')

        # noises
        self.z = fluid.layers.data(shape=[bs, self.z_dim], name='z')
        self.visual_z = fluid.layers.data(shape=[self.visual_num, self.z_dim], name='visual_z')


        d_program = fluid.Program()
        g_program = fluid.Program()
        c_program = fluid.Program()

        with fluid.program_guard(d_program):

            D_real, D_real_logits, _ = self.D(self.inputs, self.y, is_test=False)
            G_train = self.G(self.z, self.y, is_test=False)
            D_fake, D_fake_logits, _ = self.D(G_train, self.y, is_test=False)
            D_cla, D_cla_logits = self.C(self.unlabelled_inputs, is_test=False)


            ones_real = fluid.layers.fill_constant_batch_size_like(D_real, shape=[-1, 1], dtype='float32', value=1)
            zeros_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
            zeros_cla = fluid.layers.fill_constant_batch_size_like(D_cla, shape=[-1, 1], dtype='float32', value=1)

            ce_real = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_real_logits, label=ones_real)
            ce_fake = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_fake_logits, label=zeros_fake)
            ce_cla = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_cla_logits, label=zeros_cla)

            d_loss_real = fluid.layers.reduce_mean(ce_real)
            d_loss_fake = (1 - alpha) * fluid.layers.reduce_mean(ce_fake)
            d_loss_cla = alpha * fluid.layers.reduce_mean(ce_cla)
            self.d_loss = d_loss_real + d_loss_fake + d_loss_cla

        with fluid.program_guard(c_program):
            G_train = self.G(self.z, self.y, is_test=False)
            D_fake, D_fake_logits, _ = self.D(G_train, self.y)

            ones_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
            ce_fake_g = fluid.layers.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=ones_fake)
            self.g_loss = (1 - alpha) * fluid.layers.reduce_mean(ce_fake_g)

            C_real_logits = self.C(self.inputs, is_test=False)
            R_L = fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(label=self.y, logits=C_real_logits))

            # output of D for unlabelled imagesc
            Y_c = self.C(self.unlabelled_inputs, is_test=False)
            # D_cla, D_cla_logits, _ = self.D(self.unlabelled_inputs, Y_c, is_test=False)

            # output of C for fake images
            C_fake_logits = self.C(G_train, is_test=False)
            R_P = fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(label=self.y, logits=C_fake_logits))

            max_c = fluid.layers.cast(fluid.layers.argmax(Y_c, axis=1), dtype='float32')
            c_loss_dis = fluid.layers.reduce_mean(max_c * fluid.layers.softmax_with_cross_entropy(logits=D_cla_logits, label=tf.ones_like(D_cla)))
            # self.c_loss = alpha * c_loss_dis + R_L + self.alpha_p*R_P

            # R_UL = self.unsup_weight * tf.reduce_mean(tf.squared_difference(Y_c, self.unlabelled_inputs_y))
            self.c_loss = alpha_cla_adv * alpha * c_loss_dis + R_L + self.alpha_p*R_P


        with fluid.program_guard(g_program):
            G_train = self.G(self.z, self.y, is_test=False)
            D_fake, D_fake_logits, _ = self.D(G_train, self.y, is_test=False)

            ones_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
            ce_fake_g = fluid.layers.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=ones_fake)
            self.g_loss = (1 - alpha) * fluid.layers.reduce_mean(ce_fake_g)







