#-*- coding=utf-8 -*-

import cifar10
from paddleops import *
import time
import paddle.fluid as fluid
import paddle
from utils import *

class triple_gan(object):
    def __init__(self, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name, n, gan_lr, cla_lr, result_dir, log_dir):
        self.dataset_name = dataset_name
        # self.checkpoint_dir = checkpoint_dir
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


            self.gan_lr = gan_lr
            self.cla_lr = cla_lr
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

    def D(self, x, y_, name='Discriminator', is_test=False, reuse=False):
        with fluid.unique_name.guard(name+'_'):
            print("xshape: ",x.shape)
            x = dropout(x, dropout_prob=0.2, is_test=False)
            # x = reshape(x, [-1, 32, 32, 3])--->[-1,3,32,32]
            y = reshape(y_, [-1, self.y_dim, 1, 1]) #ten classes
            x = conv_cond_concat(x, y)
            #weight norm in paddlepaddle has finished
            print("convbefore!!!!!!!:",x)
            x = conv2d(x, num_filters=32, filter_size=3, param_attr=wn(), act='lrelu', reuse=reuse)
            x = conv_cond_concat(x, y)
            x = conv2d(x, num_filters=32, filter_size=3, stride=2, param_attr=wn(), act='lrelu', reuse=reuse)
            x = dropout(x, dropout_prob=0.2)
            x = conv_cond_concat(x, y)

            x = conv2d(x, num_filters=64, filter_size=3, param_attr=wn(), act='lrelu', reuse=reuse)
            x = conv_cond_concat(x, y)
            x = conv2d(x, num_filters=64, filter_size=3, stride=2, param_attr=wn(), act='lrelu', reuse=reuse)
            x = dropout(x, dropout_prob=0.2)
            x = conv_cond_concat(x, y)

            x = conv2d(x, num_filters=128, filter_size=3, param_attr=wn(), act='lrelu', reuse=reuse)
            x = conv_cond_concat(x, y)
            x = conv2d(x, num_filters=128, filter_size=3, param_attr=wn(), act='lrelu', reuse=reuse)
            x = conv_cond_concat(x, y)
            
            x = Global_Average_Pooling(x)
            x = flatten(x)
            x = concat(x, y_)
            #IcGAN 每一层都要concat一下

            # MLP??
            x_logit = fc(x, 1)
            out = sigmoid(x_logit)

            return out, x_logit, x


    def G(self, z, y, name='Generator', is_test=False, reuse=False):
        with fluid.unique_name.guard(name+'_'):
            zy = concat(z, y)
            print("zy_concat: ",zy)
            zy = fc(zy, 4*4*512, act='relu')
            zy = bn(zy, act='relu')
            zy = reshape(zy, [-1, 512, 4, 4])
            y = reshape(y, [-1, self.y_dim, 1, 1])
            zy = conv_cond_concat(zy, y)
            print("zy_before_decov1: ",zy)
            zy = deconv(zy, num_filters=256, output_size=8, stride=2, padding=1, act='relu', reuse=reuse)
            print("zy_after_decov1: ",zy)
            zy = bn(zy, act='relu')

            zy = conv_cond_concat(zy, y)
            zy = deconv(zy, num_filters=128, output_size=16, stride=2, padding=1, act='relu', reuse=reuse)
            print("zy_after_decov2: ",zy)
            zy = bn(zy, act='relu')

            zy = conv_cond_concat(zy, y)
            zy = deconv(zy, num_filters=3, output_size=32, stride=2, padding=1, act='tanh', reuse=reuse)
            print("zy_after_decov3: ",zy)
            # zy = reshape(zy, shape=[-1, self.input_height, self.input_width, self.c_dim])
            return zy

    def C(self, x, name='Classifier', is_test=False, reuse=False):
        x = gaussian_noise_layer(x, std=0.15)
        x = conv2d(x, num_filters=128, filter_size=3, act='lrelu', param_attr=wn(), reuse=reuse)
        x = conv2d(x, num_filters=128, filter_size=3, act='lrelu', param_attr=wn(), reuse=reuse)
        x = conv2d(x, num_filters=128, filter_size=3, act='lrelu', param_attr=wn(), reuse=reuse)
        x = max_pooling(x , pool_size=2)
        x = dropout(x, dropout_prob=0.5)

        x = conv2d(x, num_filters=256, filter_size=3, act='lrelu', param_attr=wn(), reuse=reuse)
        x = conv2d(x, num_filters=256, filter_size=3, act='lrelu', param_attr=wn(), reuse=reuse)
        x = conv2d(x, num_filters=256, filter_size=3, act='lrelu', param_attr=wn(), reuse=reuse)
        x = max_pooling(x , pool_size=2)
        x = dropout(x, dropout_prob=0.5)

        x = conv2d(x, num_filters=512, filter_size=3, act='lrelu', param_attr=wn(), reuse=reuse)
        print("C_conv2d7 ",x)
        x = nin(x, 256, param_attr=wn(), act='lrelu', reuse=reuse)
        x = nin(x, 128, param_attr=wn(), act='lrelu', reuse=reuse)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = fc(x, 10, param_attr=wn(), reuse=reuse)
        out = softmax(x)

        return x, out

    def loss(self, x, label):
        return fluid.layers.mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(
                x=x, label=label))
    
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def train(self):
        image_dims = [self.c_dim, self.input_height, self.input_width]
        bs = self.batch_size
        unlabel_bs = self.unlabelled_batch_size
        test_bs = self.test_batch_size
        alpha = self.alpha
        alpha_cla_adv = self.alpha_cla_adv

        def declare_data(self):
            # images
            # self.inputs = fluid.layers.data(shape=[bs] + image_dims, name='real_images')
            self.inputs = fluid.layers.data(shape=image_dims, name='real_images')
            print("self.inputs: ", self.inputs)
            # self.unlabelled_inputs = fluid.layers.data(shape=[unlabel_bs] + image_dims, name='unlabelled_images')
            self.unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')
            # self.test_inputs = fluid.layers.data(shape=[test_bs] + image_dims, name='test_images')
            self.test_inputs = fluid.layers.data(shape=image_dims, name='test_images')#nouse
            # labels
            # self.y = fluid.layers.data(shape=[bs, self.y_dim], name='y')
            self.y = fluid.layers.data(shape=[self.y_dim], name='y')
            # self.unlabelled_inputs_y = fluid.layers.data(shape=[unlabel_bs, self.y_dim], name='unlabelled_images_y')
            self.unlabelled_inputs_y = fluid.layers.data(shape=[self.y_dim], name='unlabelled_images_y')#nouse
            # self.test_label = fluid.layers.data(shape=[test_bs, self.y_dim], name='test_label')
            self.test_label = fluid.layers.data(shape=[self.y_dim], name='test_label')#nouse
            # self.visual_y = fluid.layers.data(shape=[self.visual_num, self.y_dim], name='visual_y')
            self.visual_y = fluid.layers.data(shape=[self.y_dim], name='visual_y')

            # noises
            # self.z = fluid.layers.data(shape=[bs, self.z_dim], name='z')
            self.z = fluid.layers.data(shape=[self.z_dim], name='z')
            # self.visual_z = fluid.layers.data(shape=[self.visual_num, self.z_dim], name='visual_z')
            self.visual_z = fluid.layers.data(shape=[self.z_dim], name='visual_z')

        declare_data(self)
        d_program = fluid.Program()
        g_program = fluid.Program()
        c_program = fluid.Program()

        with fluid.program_guard(d_program):
            # declare_data(self)
            self.inputs = fluid.layers.data(shape=image_dims, name='real_images')
            self.y = fluid.layers.data(shape=[self.y_dim], name='y')
            self.z = fluid.layers.data(shape=[self.z_dim], name='z')
            self.unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')

            print(self.inputs)
            D_real, D_real_logits, _ = self.D(self.inputs, self.y, is_test=False)
            G_train = self.G(self.z, self.y, is_test=False)
            print(G_train)
            D_fake, D_fake_logits, _ = self.D(G_train, self.y, is_test=False, reuse=True)
            D_cla, D_cla_logits = self.C(self.unlabelled_inputs, is_test=False)


            # ones_real = fluid.layers.fill_constant_batch_size_like(D_real, shape=[-1, 1], dtype='float32', value=1)
            # zeros_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
            # zeros_cla = fluid.layers.fill_constant_batch_size_like(D_cla, shape=[-1, 1], dtype='float32', value=1)
            ones_real = ones(D_real.shape)
            zeros_fake = zeros(D_fake.shape)
            zeros_cla = zeros(D_cla.shape)

            ce_real = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_real_logits, label=ones_real)
            ce_fake = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_fake_logits, label=zeros_fake)
            ce_cla = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_cla_logits, label=zeros_cla)

            d_loss_real = fluid.layers.reduce_mean(ce_real)
            d_loss_fake = (1 - alpha) * fluid.layers.reduce_mean(ce_fake)
            d_loss_cla = alpha * fluid.layers.reduce_mean(ce_cla)
            self.d_loss = d_loss_real + d_loss_fake + d_loss_cla

        with fluid.program_guard(c_program):
            # declare_data(self)
            self.inputs = fluid.layers.data(shape=image_dims, name='real_images')
            self.y = fluid.layers.data(shape=[self.y_dim], name='y')
            self.z = fluid.layers.data(shape=[self.z_dim], name='z')
            self.unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')


            G_train = self.G(self.z, self.y, is_test=False, reuse=True)
            # D_fake, D_fake_logits, _ = self.D(G_train, self.y)

            # ones_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
            # ce_fake_g = fluid.layers.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=ones_fake)
            # self.g_loss = (1 - alpha) * fluid.layers.reduce_mean(ce_fake_g)
            
            C_real_logits = self.C(self.inputs, is_test=False)
            R_L = fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(label=self.y, logits=C_real_logits))

            # output of D for unlabelled imagesc
            Y_c = self.C(self.unlabelled_inputs, is_test=False, reuse=True)
            D_cla, D_cla_logits, _ = self.D(self.unlabelled_inputs, Y_c, is_test=False, reuse=True)
            ones_D_cla = fluid.layers.fill_constant_batch_size_like(D_cla, shape=[-1, 1], dtype='float32', value=1)


            # output of C for fake images
            C_fake_logits = self.C(G_train, is_test=False, reuse=True)
            R_P = fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(label=self.y, logits=C_fake_logits))

            max_c = fluid.layers.cast(fluid.layers.argmax(Y_c, axis=1), dtype='float32')
            c_loss_dis = fluid.layers.reduce_mean(max_c * fluid.layers.softmax_with_cross_entropy(logits=D_cla_logits, label=ones_D_cla))
            # self.c_loss = alpha * c_loss_dis + R_L + self.alpha_p*R_P

            # R_UL = self.unsup_weight * tf.reduce_mean(tf.squared_difference(Y_c, self.unlabelled_inputs_y))
            self.c_loss = alpha_cla_adv * alpha * c_loss_dis + R_L + self.alpha_p*R_P


        with fluid.program_guard(g_program):
            # declare_data(self)
            self.inputs = fluid.layers.data(shape=image_dims, name='real_images')
            self.y = fluid.layers.data(shape=[self.y_dim], name='y')
            self.z = fluid.layers.data(shape=[self.z_dim], name='z')
            # self.unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')
            G_train = self.G(self.z, self.y, is_test=False, reuse=True)
            self.infer_program = g_program.clone(for_test=True)
            D_fake, D_fake_logits, _ = self.D(G_train, self.y, is_test=False, reuse=True)

            # ones_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
            ones_fake = ones(D_fake.shape)
            ce_fake_g = fluid.layers.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=ones_fake)
            self.g_loss = (1 - alpha) * fluid.layers.reduce_mean(ce_fake_g)


        fluid.optimizer.Adam(self.gan_lr, beta1=self.GAN_beta1).minimize(loss=self.d_loss)

        c_parameters = [p.name for p in c_program.global_block().all_parameters()]
        fluid.optimizer.Adam(self.gan_lr, beta1=self.GAN_beta1).minimize(loss=self.c_loss, parameter_list=c_parameters)

        g_parameters = [p.name for p in g_program.global_block().all_parameters()]
        fluid.optimizer.Adam(self.cla_lr, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(loss=self.g_loss, parameter_list=g_parameters)

        place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
        self.exe = fluid.Executor(place)
        self.exe.run(fluid.default_startup_program())

        start_epoch = 0

        gan_lr = self.learning_rate
        cla_lr = self.cla_learning_rate

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.visual_num, self.z_dim))
        self.test_codes = self.data_y[0:self.visual_num]
        train_reader = paddle.batch(
            paddle.dataset.cifar.train10(), batch_size=self.batch_size)

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch >= self.decay_epoch :
                gan_lr *= 0.995
                cla_lr *= 0.99
                print("**** learning rate DECAY ****")
                print(gan_lr)
                print(cla_lr)

            if epoch >= self.apply_epoch :
                alpha_p = self.apply_alpha_p
            else :
                alpha_p = self.init_alpha_p

            # rampup_value = rampup(epoch - 1)
            # unsup_weight = rampup_value * 100.0 if epoch > 1 else 0

            
            epoch_d_loss = []
            epoch_c_loss = []
            epoch_g_loss = []






        # get batch data
            for idx in range(0, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size : (idx + 1) * self.batch_size]

                batch_unlabelled_images = self.unlabelled_X[idx * self.unlabelled_batch_size : (idx + 1) * self.unlabelled_batch_size]
                batch_unlabelled_images_y = self.unlabelled_y[idx * self.unlabelled_batch_size : (idx + 1) * self.unlabelled_batch_size]

                batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                feed_dict = {
                    self.inputs: batch_images, self.y: batch_codes,
                    self.unlabelled_inputs: batch_unlabelled_images,
                    # self.unlabelled_inputs_y: batch_unlabelled_images_y,
                    self.z: batch_z, self.alpha_p: alpha_p,
                    self.gan_lr: gan_lr, self.cla_lr: cla_lr
                    # self.unsup_weight : unsup_weight
                }
                # update D network
                # _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=feed_dict)
                # self.writer.add_summary(summary_str, counter)
                d_loss = self.exe.run(d_program, feed=feed_dict, fetch_list={self.d_loss})
                g_loss = self.exe.run(g_program, feed=feed_dict, fetch_list={self.g_loss})
                c_loss = self.exe.run(c_program, feed=feed_dict, fetch_list={self.c_loss})


                # # update G network
                # _, summary_str_g, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=feed_dict)
                # self.writer.add_summary(summary_str_g, counter)

                # # update C network
                # _, summary_str_c, c_loss = self.sess.run([self.c_optim, self.c_sum, self.c_loss], feed_dict=feed_dict)
                # self.writer.add_summary(summary_str_c, counter)

                # display training status
                # counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, c_loss))

                # save training results for every 100 steps
                """
                if np.mod(counter, 100) == 0:
                    samples = self.sess.run(self.infer_program,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    image_frame_dim = int(np.floor(np.sqrt(self.visual_num)))
                    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))
                """

            # classifier test
            # test_acc = 0.0

            # for idx in range(10) :
            #     test_batch_x = self.test_X[idx * self.test_batch_size : (idx+1) * self.test_batch_size]
            #     test_batch_y = self.test_y[idx * self.test_batch_size : (idx+1) * self.test_batch_size]

            #     acc_ = self.sess.run(self.accuracy, feed_dict={
            #         self.test_inputs: test_batch_x,
            #         self.test_label: test_batch_y
            #     })

            #     test_acc += acc_
            # test_acc /= 10

            # summary_test = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
            # self.writer.add_summary(summary_test, epoch)

            # line = "Epoch: [%2d], test_acc: %.4f\n" % (epoch, test_acc)
            # print(line)
            # lr = "{} {}".format(gan_lr, cla_lr)
            # with open('logs.txt', 'a') as f:
            #     f.write(line)
            # with open('lr_logs.txt', 'a') as f :
            #     f.write(lr+'\n')

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            # self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

            # save model for final step
        # self.save(self.checkpoint_dir, counter)
    def visualize_results(self, epoch):
        # tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(self.visual_num)))
        z_sample = np.random.uniform(-1, 1, size=(self.visual_num, self.z_dim))
        self.visual_z = fluid.layers.data(shape=[self.z_dim], name='visual_z')
        """ random noise, random discrete code, fixed continuous code """
        y = np.random.choice(self.len_discrete_code, self.visual_num)
        # Generated 10 labels with batch_size
        y_one_hot = np.zeros((self.visual_num, self.y_dim))
        y_one_hot[np.arange(self.visual_num), y] = 1

        samples = self.exe.run(self.infer_program, feed={self.visual_z: z_sample, self.visual_y: y_one_hot})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir + '/all_classes') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to self.batch_size

        np.random.seed()
        si = np.random.choice(self.visual_num, n_styles)

        for l in range(self.len_discrete_code):
            y = np.zeros(self.visual_num, dtype=np.int64) + l
            y_one_hot = np.zeros((self.visual_num, self.y_dim))
            y_one_hot[np.arange(self.visual_num), y] = 1

            samples = self.exe.run(self.infer_program, feed={self.visual_z: z_sample, self.visual_y: y_one_hot})
            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        check_folder(
                            self.result_dir + '/' + self.model_dir + '/class_%d' % l) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.len_discrete_code):
                canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, self.len_discrete_code],
                    check_folder(
                        self.result_dir + '/' + self.model_dir + '/all_classes_style_by_style') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')






 



if __name__ == "__main__":


    epoch = 1000
    batch_size = 20
    unlabel_batch_size = 250
    z_dim = 100
    dataset_name = 0
    n = 4000
    gan_lr = 3e-4
    cla_lr = 3e-3
    # checkpoint_dir = 0
    dataset_name = 'cifar10'
    result_dir = './result/'
    check_folder(result_dir)
    log_dir = './log/'
    check_folder(log_dir)
    GAN = triple_gan(epoch, batch_size, unlabel_batch_size, z_dim, dataset_name, n, gan_lr, cla_lr, result_dir, log_dir)
    GAN.train()













        







