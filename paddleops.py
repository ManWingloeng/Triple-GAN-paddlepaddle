import paddle
import paddle.fluid as fluid
import inspect
import numpy as np
import math

def get_parent_function_name():
    return inspect.stack()[2][3] + '.' + inspect.stack()[1][3] + '.' + str(
        inspect.stack()[2][2]) + '.'

def cal_padding(img_size, stride, filter_size, dilation=1):
    """Calculate padding size."""
    valid_filter_size = dilation * (filter_size - 1) + 1
    if img_size % stride == 0:
        out_size = max(filter_size - stride, 0)
    else:
        out_size = max(filter_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2

def bn(x, name=None, act='relu'):
    if name is None:
        name = get_parent_function_name()
    #return fluid.layers.leaky_relu(x)
    return fluid.layers.batch_norm(
        x,
        param_attr=name + '1',
        bias_attr=name + '2',
        moving_mean_name=name + '3',
        moving_variance_name=name + '4',
        name=name,
        act=act)

def wn() :
    return fluid.WeightNormParamAttr(initializer=fluid.initializer.MSRAInitializer())


def conv(x, num_filters, name=None, act=None):
    if name is None:
        name = get_parent_function_name()
    return fluid.nets.simple_img_conv_pool(
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        pool_stride=2,
        param_attr=name + 'w',
        bias_attr=name + 'b',
        act=act)


def fc(x, num_filters, param_attr=None, name=None, act=None):
    if name is None:
        name = get_parent_function_name()
    return fluid.layers.fc(input=x,
                           size=num_filters,
                           act=act,
                           param_attr=name + '_w',
                           bias_attr=name + '_b')

def sigmoid(x):
    return fluid.layers.sigmoid(x)

def ones(shape, dtype='float32'):
    return fluid.layers.ones(shape, dtype)

def zeros(shape, dtype='float32'):
    return fluid.layers.zeros(shape, dtype)

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    # print("x: ",x)
    # print("y: ",y)
    # one = fluid.layers.fill_constant_batch_size_like(
    #     x, [-1, x.shape[1], x.shape[2], y.shape[3]], "float32", value=1.0)
    return fluid.layers.concat([x, ones(shape=[x.shape[0], y.shape[1], x.shape[2], x.shape[3]]) * y], 1)
    # print("one: ",one)
    # y_one = one * y
    # print("y_one: ",y_one)

    
        # return fluid.layers.concat([x, y_one], axis=3)


def concat(x, y, axis=1):
    return fluid.layers.concat([x, y], axis=axis)

def flatten(x, axis=1):
    return fluid.layers.flatten(x, axis=axis)

def conv2d( input,
            num_filters,
            filter_size,
            stride=1,
            param_attr=None,
            padding='SAME',
            act=None,
            leak=0.2,
            name=None,
            ):
    if name is None:
        name = get_parent_function_name()
    if param_attr is None:
        param_attr = name+'_w'
    """Wrapper for conv2d op to support VALID and SAME padding mode."""
    need_crop = False
    if padding == 'SAME':
        top_padding, bottom_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        print("top_padding, bottom_padding:",(top_padding, bottom_padding))
        left_padding, right_padding = cal_padding(input.shape[2], stride,
                                                  filter_size)
        height_padding = bottom_padding
        width_padding = right_padding
        if top_padding != bottom_padding or left_padding != right_padding:
            height_padding = top_padding + stride
            width_padding = left_padding + stride
            need_crop = True
    else:
        height_padding = 0
        width_padding = 0

    padding = [height_padding, width_padding]
    bias_attr = fluid.ParamAttr(
        name=name + "_b", initializer=fluid.initializer.Constant(0.0))

    # if need_crop:
    #     conv = fluid.layers.crop(
    #         conv,
    #         shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
    #         offsets=(0, 0, 1, 1))
    # if norm:
    #     conv = instance_norm(input=conv, name=name + "_norm")
    if act=='lrelu':
        conv = fluid.layers.conv2d(
            input,
            num_filters,
            filter_size,
            name=name,
            stride=stride,
            padding=padding,
            use_cudnn=False,
            param_attr=param_attr,
            bias_attr=bias_attr)
        if need_crop:
            conv = fluid.layers.crop(
                conv,
                shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
                offsets=(0, 0, 1, 1))
        conv = fluid.layers.leaky_relu(conv, alpha=leak)
    else:
        conv = fluid.layers.conv2d(
            input,
            num_filters,
            filter_size,
            name=name,
            stride=stride,
            padding=padding,
            use_cudnn=False,
            act=act,
            param_attr=param_attr,
            bias_attr=bias_attr)
        if need_crop:
            conv = fluid.layers.crop(
                conv,
                shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
                offsets=(0, 0, 1, 1))
    return conv


def deconv( x,
            num_filters,
            name=None,
            filter_size=None,
            stride=2,
            #    dilation=1,
            #    padding=2,
            padding = "SAME",    
            output_size=None,
            param_attr=None,
            act=None):
    if name is None:
        name = get_parent_function_name()
    if param_attr is None:
        param_attr=name + 'w'
    need_crop = False
    if padding == "SAME":
        top_padding, bottom_padding = cal_padding(x.shape[2], stride,
                                                  filter_size)
        print("(x.shape[2], stride,num_filters):",(x.shape[2], stride, filter_size))
        print("top_padding, bottom_padding:",(top_padding, bottom_padding))
        left_padding, right_padding = cal_padding(x.shape[2], stride,
                                                  filter_size)
        height_padding = bottom_padding
        width_padding = right_padding
        if top_padding != bottom_padding or left_padding != right_padding:
            height_padding = top_padding + stride
            width_padding = left_padding + stride
            need_crop = True
        padding = [height_padding, width_padding]
    print("padding",padding)

    conv = fluid.layers.conv2d_transpose(
        input=x,
        param_attr=param_attr,
        bias_attr=name + 'b',
        num_filters=num_filters,
        output_size=output_size,
        filter_size=filter_size,
        stride=stride,
        # dilation=dilation,
        padding=padding,
        act=act)
    if need_crop:
        conv = fluid.layers.crop(
            conv,
            shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
            offsets=(0, 0, 0, 0))

    return conv

# def deconv( input,
#             num_filters,
#             filter_size,
#             stride=1,
#             param_attr=None,
#             padding="SAME",
#             act=None,
#             leak=0.2,
#             name="deconv",
#             ):
#     if name is None:
#         name = get_parent_function_name()
#     # if param_attr is None:
#     #     param_attr = name+'_w'
#     """Wrapper for conv2d op to support VALID and SAME padding mode."""
#     need_crop = False
#     if padding == "SAME":
#         top_padding, bottom_padding = cal_padding(input.shape[2], stride,
#                                                   filter_size)
#         left_padding, right_padding = cal_padding(input.shape[2], stride,
#                                                   filter_size)
#         height_padding = bottom_padding
#         width_padding = right_padding
#         if top_padding != bottom_padding or left_padding != right_padding:
#             height_padding = top_padding + stride
#             width_padding = left_padding + stride
#             need_crop = True
#     else:
#         height_padding = 0
#         width_padding = 0

#     padding = [height_padding, width_padding]
#     # bias_attr = fluid.ParamAttr(
#     #     name=name + "_b", initializer=fluid.initializer.Constant(0.0))

#     # if need_crop:
#     #     conv = fluid.layers.crop(
#     #         conv,
#     #         shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
#     #         offsets=(0, 0, 1, 1))
#     # if norm:
#     #     conv = instance_norm(input=conv, name=name + "_norm")
#     if act=='lrelu':
#         conv = fluid.layers.conv2d_transpose(
#             input,
#             num_filters,
#             filter_size,
#             name=name,
#             stride=stride,
#             padding=padding,
#             use_cudnn=False,)
#             # param_attr=param_attr,)
#             # bias_attr=bias_attr)
#         if need_crop:
#             conv = fluid.layers.crop(
#                 conv,
#                 shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
#                 offsets=(0, 0, 1, 1))
#         conv = fluid.layers.leaky_relu(conv, alpha=leak)
#     else:
#         conv = fluid.layers.conv2d_transpose(
#             input,
#             num_filters,
#             filter_size,
#             name=name,
#             stride=stride,
#             padding=padding,
#             use_cudnn=False,
#             act=act,)
#             # param_attr=param_attr,)
#             # bias_attr=bias_attr)
#         if need_crop:
#             conv = fluid.layers.crop(
#                 conv,
#                 shape=(-1, conv.shape[1], conv.shape[2] - 1, conv.shape[3] - 1),
#                 offsets=(0, 0, 1, 1))
#     return conv

def lrelu(x, leak=0.2, name="lrelu"):
    return fluid.layers.leaky_relu(x, alpha=leak)

def dropout(x, dropout_prob, is_test=False, seed=None, name="dropout"):
    return fluid.layers.dropout(x, dropout_prob, is_test, seed, name)

# def rampup(epoch):
#     if epoch < 80:
#         p = max(0.0, float(epoch)) / float(80)
#         p = 1.0 - p
#         return math.exp(-p*p*5.0)
#     else:
#         return 1.0

def rampdown(epoch):
    if epoch >= (300 - 50):
        ep = (epoch - (300 - 50)) * 0.5
        return math.exp(-(ep * ep) / 50)
    else:
        return 1.0


def reshape(x, shape):
    return fluid.layers.reshape(x, shape)


def max_pooling(x, pool_size=-1, stride=1, padding=0):
    return fluid.layers.pool2d(x, pool_size=pool_size, stride=stride, pool_padding=padding)

def Global_Average_Pooling(x):
    return fluid.layers.pool2d(x, pool_type='avg', global_pooling=True)

def gaussian_noise_layer(x, std=0.15):
    noise = fluid.layers.gaussian_random(shape=fluid.layers.shape(x), mean=0.0, std=std)
    return x + noise

def nin(x, num_units, name='nin', param_attr=None, act=None):
    """ a network in network layer (1x1 CONV) """
    s = list(map(int, x.get_shape()))
    x = reshape(x, [np.prod(s[:-1]),s[-1]])
    if act=='lrelu':
        x = fc(x, num_units, name=name, param_attr=param_attr)
        x = lrelu(x)
    else:
        x = fc(x, num_units, name=name, param_attr=param_attr, act=act)
    return reshape(x, s[:-1]+[num_units])

def softmax(x):
    return fluid.layers.softmax(x)