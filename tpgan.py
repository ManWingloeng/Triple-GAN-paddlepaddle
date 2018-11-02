#-*- coding=utf-8 -*-

import cifar10
from paddleops import *
import time
import paddle.fluid as fluid
import paddle
from utils import *

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

# def __init__( epoch, batch_size, unlabel_batch_size, z_dim, dataset_name, n, gan_lr, cla_lr, result_dir, log_dir):
dataset_name = dataset_name
# checkpoint_dir = checkpoint_dir
result_dir = result_dir
log_dir = log_dir
epoch = epoch
batch_size = batch_size
unlabelled_batch_size = unlabel_batch_size
test_batch_size = 1000
model_name = "TripleGAN"     # name for checkpoint

input_height = 32
input_width = 32
output_height = 32
output_width = 32

z_dim = z_dim
y_dim = 10
c_dim = 3


gan_lr = gan_lr
cla_lr = cla_lr
learning_rate = gan_lr # 3e-4, 1e-3
cla_learning_rate = cla_lr # 3e-3, 1e-2 ?
GAN_beta1 = 0.5
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
alpha = 0.5
alpha_cla_adv = 0.01
init_alpha_p = 0.0 # 0.1, 0.03
apply_alpha_p = 0.1
alpha_p = 0.1 ##??

apply_epoch = 200 # 200, 300
decay_epoch = 50

sample_num = 64
visual_num = 100
len_discrete_code = 10

data_X, data_y, unlabelled_X, unlabelled_y, test_X, test_y = cifar10.prepare_data(n) # trainX, trainY, testX, testY

num_batches = len(data_X) // batch_size

def D( x, y_, name='D', is_test=False, reuse=False):
    with fluid.unique_name.guard(name+'_'):


        # print("xshape: ",x.shape)
        # x = dropout(x, dropout_prob=0.2, is_test=False)
        # # x = reshape(x, [-1, 32, 32, 3])--->[-1,3,32,32]
        # y = reshape(y_, [-1, y_dim, 1, 1]) #ten classes
        # x = conv_cond_concat(x, y)
        # #weight norm in paddlepaddle has finished
        # print("convbefore!!!!!!!:",x)
        # x = conv2d(x, num_filters=32, filter_size=3, 
        #                 param_attr=None, act='lrelu', reuse=reuse)
        # x = conv_cond_concat(x, y)
        # x = conv2d(x, num_filters=32, filter_size=3, stride=2, 
        #                 param_attr=None, act='lrelu', reuse=reuse)
        # print("D,con2d_2 shape:",x.shape)
        # x = dropout(x, dropout_prob=0.2)
        # x = conv_cond_concat(x, y)

        # x = conv2d(x, num_filters=64, filter_size=3, param_attr=None, act='lrelu', reuse=reuse)
        # x = conv_cond_concat(x, y)
        # x = conv2d(x, num_filters=64, filter_size=3, stride=2, param_attr=None, act='lrelu', reuse=reuse)
        # x = dropout(x, dropout_prob=0.2)
        # x = conv_cond_concat(x, y)

        # x = conv2d(x, num_filters=128, filter_size=3, param_attr=None, act='lrelu', reuse=reuse)
        # x = conv_cond_concat(x, y)
        # x = conv2d(x, num_filters=128, filter_size=3, param_attr=None, act='lrelu', reuse=reuse)
        # x = conv_cond_concat(x, y)
        
        # x = Global_Average_Pooling(x)
        # x = flatten(x)



        # x = concat(x, y_)
        #IcGAN 每一层都要concat一下

        # MLP??
        x_logit = fc(x, 1)
        out = sigmoid(x_logit)

        return out, x_logit, x


def G( z, y, name='G', is_test=False, reuse=False):
    with fluid.unique_name.guard(name+'_'):
        zy = concat(z, y)
        print("zy_concat: ",zy)
        zy = fc(zy, 4*4*512, act='relu')
        zy = bn(zy, act='relu')
        zy = reshape(zy, [-1, 512, 4, 4])
        y = reshape(y, [-1, y_dim, 1, 1])
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
        # zy = reshape(zy, shape=[-1, input_height, input_width, c_dim])
        return zy

def C( x, name='C', is_test=False, reuse=False):
    with fluid.unique_name.guard(name+'_'):
        x = gaussian_noise_layer(x, std=0.15)
        print("C input shape: ",x.shape)
        x = conv2d(x, num_filters=128, filter_size=3, act='lrelu', param_attr=None, reuse=reuse)
        x = conv2d(x, num_filters=128, filter_size=3, act='lrelu', param_attr=None, reuse=reuse)
        x = conv2d(x, num_filters=128, filter_size=3, act='lrelu', param_attr=None, reuse=reuse)
        print("conv2d C shape: ",x.shape)
        x = max_pooling(x , pool_size=2, pool_stride=2)
        print("maxpool C shape:",x.shape)
        x = dropout (x, dropout_prob=0.5)

        x = conv2d(x, num_filters=256, filter_size=3, act='lrelu', param_attr=None, reuse=reuse)
        x = conv2d(x, num_filters=256, filter_size=3, act='lrelu', param_attr=None, reuse=reuse)
        x = conv2d(x, num_filters=256, filter_size=3, act='lrelu', param_attr=None, reuse=reuse)
        x = max_pooling(x , pool_size=2, pool_stride=2)
        x = dropout(x, dropout_prob=0.5)

        x = conv2d(x, num_filters=512, filter_size=3, act='lrelu', param_attr=None, reuse=reuse)
        print("C_conv2d7 ",x)
        x = nin(x, 256, name='nin1', param_attr=None, act='lrelu', reuse=reuse)
        x = nin(x, 128, name='nin2', param_attr=None, act='lrelu', reuse=reuse)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = fc(x, 10, param_attr=None, reuse=reuse)
        out = softmax(x)

        return x, out

def loss( x, label):
    return fluid.layers.mean(
        fluid.layers.sigmoid_cross_entropy_with_logits(
            x=x, label=label))

@property
def model_dir():
    return "{}_{}_{}_{}".format(
        model_name, dataset_name,
        batch_size, z_dim)


def get_params( program, prefix):
    all_params = program.global_block().all_parameters()
    return [t.name for t in all_params if t.name.startswith(prefix)]


def train():
    image_dims = [c_dim, input_height, input_width]
    bs = batch_size
    unlabel_bs = unlabelled_batch_size
    test_bs = test_batch_size
    alpha = 0.5
    alpha_cla_adv = 0.01

    def declare_data():
        # images
        # inputs = fluid.layers.data(shape=[bs] + image_dims, name='real_images')
        inputs = fluid.layers.data(shape=image_dims, name='real_images')
        print("inputs: ", inputs)
        # unlabelled_inputs = fluid.layers.data(shape=[unlabel_bs] + image_dims, name='unlabelled_images')
        unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')
        # test_inputs = fluid.layers.data(shape=[test_bs] + image_dims, name='test_images')
        test_inputs = fluid.layers.data(shape=image_dims, name='test_images')#nouse
        # labels
        # y = fluid.layers.data(shape=[bs, y_dim], name='y')
        y = fluid.layers.data(shape=[y_dim], name='y')
        # unlabelled_inputs_y = fluid.layers.data(shape=[unlabel_bs, y_dim], name='unlabelled_images_y')
        unlabelled_inputs_y = fluid.layers.data(shape=[y_dim], name='unlabelled_images_y')#nouse
        # test_label = fluid.layers.data(shape=[test_bs, y_dim], name='test_label')
        test_label = fluid.layers.data(shape=[y_dim], name='test_label')#nouse
        # visual_y = fluid.layers.data(shape=[visual_num, y_dim], name='visual_y')
        visual_y = fluid.layers.data(shape=[y_dim], name='visual_y')

        # noises
        # z = fluid.layers.data(shape=[bs, z_dim], name='z')
        z = fluid.layers.data(shape=[z_dim], name='z')
        # visual_z = fluid.layers.data(shape=[visual_num, z_dim], name='visual_z')
        visual_z = fluid.layers.data(shape=[z_dim], name='visual_z')

    declare_data()
    d_program = fluid.Program()
    g_program = fluid.Program()
    c_program = fluid.Program()

    with fluid.program_guard(d_program):
        # declare_data()
        inputs = fluid.layers.data(shape=image_dims, name='real_images')
        y = fluid.layers.data(shape=[y_dim], name='y')
        z = fluid.layers.data(shape=[z_dim], name='z')
        unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')

        print(inputs)
        D_real, D_real_logits, _ = D(inputs, y, is_test=False)
        print("D_real: ",D_real)
        G_train = G(z, y, is_test=False)
        print(G_train)
        D_fake, D_fake_logits, _ = D(G_train, y, is_test=False, reuse=True)
        Y_c, _ = C(unlabelled_inputs, is_test=False)
        D_cla, D_cla_logits, _ = D(unlabelled_inputs, Y_c, is_test=False, reuse=True)



        ones_real = fluid.layers.fill_constant_batch_size_like(D_real, shape=[-1, 1], dtype='float32', value=1)
        zeros_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=0)
        zeros_cla = fluid.layers.fill_constant_batch_size_like(D_cla, shape=[-1, 1], dtype='float32', value=0)
        # ones_real = ones(D_real.shape)
        # ones_real.stop_gradient = True
        # zeros_fake = zeros(D_fake.shape)
        # zeros_fake.stop_gradient = True
        # zeros_cla = zeros(D_cla.shape)
        # zeros_cla.stop_gradient = True

        ce_real = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_real_logits, label=ones_real)
        ce_fake = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_fake_logits, label=zeros_fake)
        ce_cla = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_cla_logits, label=zeros_cla)

        d_loss_real = fluid.layers.reduce_mean(ce_real)
        d_loss_fake = (1 - alpha) * fluid.layers.reduce_mean(ce_fake)
        d_loss_cla = alpha * fluid.layers.reduce_mean(ce_cla)
        # d_R_F_loss = fluid.layers.elementwise_add(d_loss_real, d_loss_fake)
        d_loss = d_loss_real + d_loss_fake + d_loss_cla

        d_parameters = get_params(d_program, prefix='D')
        # d_all_params = d_program.global_block().all_parameters()
        # d_params= [t.name for t in d_all_params]
        # d_opt = fluid.optimizer.RMSPropOptimizer(learning_rate=gan_lr)
        lr = 0.0002
        d_opt = fluid.optimizer.Adam(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=[
                    100 * decay_epoch, 120 * decay_epoch,
                    140 * decay_epoch, 160 * decay_epoch,
                    180 * decay_epoch
                ],
                values=[
                    lr, lr * 0.8, lr * 0.6, lr * 0.4, lr * 0.2, lr * 0.1
                ]),
            beta1=0.5)
        # d_loss = D_fake
        d_loss = fluid.layers.reduce_mean(D_fake_logits)
        d_opt.minimize(d_loss, parameter_list=d_parameters)

    with fluid.program_guard(c_program):
        # declare_data()
        inputs = fluid.layers.data(shape=image_dims, name='real_images')
        y = fluid.layers.data(shape=[y_dim], name='y')
        z = fluid.layers.data(shape=[z_dim], name='z')
        unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')


        G_train = G(z, y, is_test=False, reuse=True)
        # D_fake, D_fake_logits, _ = D(G_train, y)

        # ones_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
        # ce_fake_g = fluid.layers.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=ones_fake)
        # g_loss = (1 - alpha) * fluid.layers.reduce_mean(ce_fake_g)
        
        C_real_logits, _ = C(inputs, is_test=False, reuse=True)
        print("C_real_logits: ",C_real_logits)
        print("y: ",y)
        R_L = fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(logits=C_real_logits, label=y, soft_label=True))

        # output of D for unlabelled imagesc
        Y_c, _ = C(unlabelled_inputs, is_test=False, reuse=True)
        D_cla, D_cla_logits, _ = D(unlabelled_inputs, Y_c, is_test=False, reuse=True)
        # ones_D_cla = fluid.layers.fill_constant_batch_size_like(D_cla, shape=[-1, 1], dtype='float32', value=1)
        ones_D_cla = ones(D_cla.shape)

        # output of C for fake images
        C_fake_logits, _ = C(G_train, is_test=False, reuse=True)
        print("C_fake_logits: ",C_fake_logits)
        print("y: ",y)
        R_P = fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(label=y, logits=C_fake_logits, soft_label=True))

        max_c = fluid.layers.cast(fluid.layers.argmax(Y_c, axis=1), dtype='float32')
        print("max_c: ", max_c)
        ce_D_cla = fluid.layers.softmax_with_cross_entropy(logits=D_cla_logits, label=ones_D_cla, soft_label=True)
        ce_D_cla = reshape(ce_D_cla,shape=[-1])
        print("ce_D_cla: ", ce_D_cla)
        c_loss_dis = fluid.layers.reduce_mean(max_c * ce_D_cla)
        # c_loss = alpha * c_loss_dis + R_L + alpha_p*R_P

        # R_UL = unsup_weight * tf.reduce_mean(tf.squared_difference(Y_c, unlabelled_inputs_y))
        c_loss = alpha_cla_adv * alpha * c_loss_dis + R_L + alpha_p*R_P


    with fluid.program_guard(g_program):
        # declare_data()
        inputs = fluid.layers.data(shape=image_dims, name='real_images')
        y = fluid.layers.data(shape=[y_dim], name='y')
        z = fluid.layers.data(shape=[z_dim], name='z')
        # unlabelled_inputs = fluid.layers.data(shape=image_dims, name='unlabelled_images')
        G_train = G(z, y, is_test=False, reuse=True)
        infer_program = g_program.clone(for_test=True)
        D_fake, D_fake_logits, _ = D(G_train, y, is_test=False, reuse=True)

        # ones_fake = fluid.layers.fill_constant_batch_size_like(D_fake, shape=[-1, 1], dtype='float32', value=1)
        ones_fake = ones(D_fake.shape)
        ce_fake_g = fluid.layers.sigmoid_cross_entropy_with_logits(x=D_fake_logits, label=ones_fake)
        g_loss = (1 - alpha) * fluid.layers.reduce_mean(ce_fake_g)

    d_parameters = get_params(d_program, prefix='D')
    d_all_params = d_program.global_block().all_parameters()
    d_params= [t.name for t in d_all_params]
    d_opt = fluid.optimizer.Adam(learning_rate=gan_lr, beta1=GAN_beta1)
    # d_opt.minimize(d_loss, parameter_list=d_parameters)
    print(d_parameters)
    print(d_params)


    c_parameters = get_params(c_program, prefix='C')
    c_opt = fluid.optimizer.Adam(learning_rate=gan_lr, beta1=GAN_beta1)
    # c_opt.minimize(loss=c_loss, parameter_list=c_parameters)
    
    print(c_parameters)
    
    
    g_parameters = get_params(g_program, prefix='G')
    g_opt = fluid.optimizer.Adam(learning_rate=cla_lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
    # g_opt.minimize(loss=g_loss, parameter_list=g_parameters)
    print(g_parameters)


    place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    start_epoch = 0

    gan_lr = learning_rate
    cla_lr = cla_learning_rate

    # graph inputs for visualize training results
    sample_z = np.random.uniform(-1, 1, size=(visual_num, z_dim))
    test_codes = data_y[0:visual_num]
    train_reader = paddle.batch(
        paddle.dataset.cifar.train10(), batch_size=batch_size)

    start_time = time.time()
    for epoch in range(start_epoch, epoch):
        if epoch >= decay_epoch :
            gan_lr *= 0.995
            cla_lr *= 0.99
            print("**** learning rate DECAY ****")
            print(gan_lr)
            print(cla_lr)

        if epoch >= apply_epoch :
            alpha_p = apply_alpha_p
        else :
            alpha_p = init_alpha_p

        # rampup_value = rampup(epoch - 1)
        # unsup_weight = rampup_value * 100.0 if epoch > 1 else 0

        
        epoch_d_loss = []
        epoch_c_loss = []
        epoch_g_loss = []






    # get batch data
        for idx in range(0, num_batches):
            batch_images = data_X[idx * batch_size : (idx + 1) * batch_size]
            batch_codes = data_y[idx * batch_size : (idx + 1) * batch_size]

            batch_unlabelled_images = unlabelled_X[idx * unlabelled_batch_size : (idx + 1) * unlabelled_batch_size]
            batch_unlabelled_images_y = unlabelled_y[idx * unlabelled_batch_size : (idx + 1) * unlabelled_batch_size]

            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

            feed_dict = {
                inputs: batch_images, y: batch_codes,
                unlabelled_inputs: batch_unlabelled_images,
                # unlabelled_inputs_y: batch_unlabelled_images_y,
                z: batch_z, alpha_p: alpha_p,
                gan_lr: gan_lr, cla_lr: cla_lr
                # unsup_weight : unsup_weight
            }
            # update D network
            # _, summary_str, d_loss = sess.run([d_optim, d_sum, d_loss], feed_dict=feed_dict)
            # writer.add_summary(summary_str, counter)
            d_loss = exe.run(d_program, feed=feed_dict, fetch_list={d_loss})
            g_loss = exe.run(g_program, feed=feed_dict, fetch_list={g_loss})
            c_loss = exe.run(c_program, feed=feed_dict, fetch_list={c_loss})


            # # update G network
            # _, summary_str_g, g_loss = sess.run([g_optim, g_sum, g_loss], feed_dict=feed_dict)
            # writer.add_summary(summary_str_g, counter)

            # # update C network
            # _, summary_str_c, c_loss = sess.run([c_optim, c_sum, c_loss], feed_dict=feed_dict)
            # writer.add_summary(summary_str_c, counter)

            # display training status
            # counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" \
                    % (epoch, idx, num_batches, time.time() - start_time, d_loss, g_loss, c_loss))

            # save training results for every 100 steps
            """
            if np.mod(counter, 100) == 0:
                samples = sess.run(infer_program,
                                        feed_dict={z: sample_z, y: test_codes})
                image_frame_dim = int(np.floor(np.sqrt(visual_num)))
                save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                            './' + check_folder(
                                result_dir + '/' + model_dir) + '/' + model_name + '_train_{:02d}_{:04d}.png'.format(
                                epoch, idx))
            """

        # classifier test
        # test_acc = 0.0

        # for idx in range(10) :
        #     test_batch_x = test_X[idx * test_batch_size : (idx+1) * test_batch_size]
        #     test_batch_y = test_y[idx * test_batch_size : (idx+1) * test_batch_size]

        #     acc_ = sess.run(accuracy, feed_dict={
        #         test_inputs: test_batch_x,
        #         test_label: test_batch_y
        #     })

        #     test_acc += acc_
        # test_acc /= 10

        # summary_test = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
        # writer.add_summary(summary_test, epoch)

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
        # save(checkpoint_dir, counter)

        # show temporal results
        # visualize_results(epoch)
            # tot_num_samples = min(sample_num, batch_size)
        image_frame_dim = int(np.floor(np.sqrt(visual_num)))
        z_sample = np.random.uniform(-1, 1, size=(visual_num, z_dim))
        visual_z = fluid.layers.data(shape=[z_dim], name='visual_z')
        """ random noise, random discrete code, fixed continuous code """
        y = np.random.choice(len_discrete_code, visual_num)
        # Generated 10 labels with batch_size
        y_one_hot = np.zeros((visual_num, y_dim))
        y_one_hot[np.arange(visual_num), y] = 1

        samples = exe.run(infer_program, feed={visual_z: z_sample, visual_y: y_one_hot})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        result_dir + '/' + model_dir + '/all_classes') + '/' + model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to batch_size

        np.random.seed()
        si = np.random.choice(visual_num, n_styles)

        for l in range(len_discrete_code):
            y = np.zeros(visual_num, dtype=np.int64) + l
            y_one_hot = np.zeros((visual_num, y_dim))
            y_one_hot[np.arange(visual_num), y] = 1

            samples = exe.run(infer_program, feed={visual_z: z_sample, visual_y: y_one_hot})
            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        check_folder(
                            result_dir + '/' + model_dir + '/class_%d' % l) + '/' + model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(len_discrete_code):
                canvas[s * len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, len_discrete_code],
                    check_folder(
                        result_dir + '/' + model_dir + '/all_classes_style_by_style') + '/' + model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')
            # save model for final step
        # save(checkpoint_dir, counter)
# def visualize_results( epoch):







 



if __name__ == "__main__":
    train()













        







