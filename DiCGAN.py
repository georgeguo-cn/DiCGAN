"""
    @Time   : 2020.01.16
    @Author : Zhiqiang Guo
    @Email  : zhiqiangguo@hust.edu.cn
    the model of DiCGAN
"""

import tensorflow as tf

tf.set_random_seed(20200115)
slim = tf.contrib.slim

class DiCGAN(object):

    def __init__(self, args, model_args, num_items, num_users):
        self.args = args
        self.model_args = model_args
        self.n_items = num_items
        self.n_users = num_users
        self.L = args.L
        self.percentage = args.percentage
        self.emb_len = args.emb_len
        self.dilations = args.dilations
        self.conv_channels = self.emb_len
        self.kernel_size = args.kernel_size
        self.drop_rate = args.drop_rate
        self.is_attention = args.is_attention
        self.causal = args.causal
        self.loss_type = args.loss_type

        self.item_embs = tf.get_variable('item_embs', [num_items, self.emb_len], initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.item_zero = tf.get_variable('item_zero', [1, self.emb_len], initializer=tf.zeros_initializer(), trainable=False)
        self.user_embs = tf.get_variable('user_embs', [num_users, self.emb_len], initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.model()

        self.saver = tf.train.Saver(max_to_keep=1)

    def fc_layer(self, input, inputDim, outputDim, activation, name, layer_id, is_train=0):

        wName = name + "_W" + str(layer_id)
        bName = name + "_b" + str(layer_id)

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            W = tf.get_variable(wName, [inputDim, outputDim],
                                initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
            b = tf.get_variable(bName, [outputDim],
                                initializer=tf.random_uniform_initializer(-0.01, 0.01))

            y = tf.matmul(input, W) + b

            L2norm = tf.nn.l2_loss(W) + tf.nn.l2_loss(b)

            if activation == "none":
                y = tf.identity(y, name="output")
            elif activation == "tanh":
                y = tf.nn.tanh(y)
            elif activation == "relu":
                y = tf.nn.relu(y)
            elif activation == "leaky_relu":
                y = tf.nn.leaky_relu(y)
            elif activation == "sigmoid":
                y = tf.nn.sigmoid(y)
            if is_train == 1:
                y = tf.nn.dropout(y, rate=self.drop_rate)
        return y, L2norm, W, b

    def layer_norm(self, x, name, epsilon=1e-8, is_train=0):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            if is_train == 1:
                trainable = True
            else:
                trainable = False
            shape = x.get_shape()
            shift = tf.get_variable('shift', [int(shape[-1])],
                                    initializer=tf.constant_initializer(0), trainable=trainable)
            scale = tf.get_variable('scale', [int(shape[-1])],
                                    initializer=tf.constant_initializer(1), trainable=trainable)

            mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

            BN = tf.nn.batch_normalization(x, mean, variance, shift, scale, epsilon)

        return BN

    def combined_static_and_dynamic_shape(self, tensor):
        static_tensor_shape = tensor.shape.as_list()
        dynamic_tensor_shape = tf.shape(tensor)
        combined_shape = []
        for index, dim in enumerate(static_tensor_shape):
            if dim is not None:
                combined_shape.append(dim)
            else:
                combined_shape.append(dynamic_tensor_shape[index])
        return combined_shape

    def conv_block_attention(self, feature_map, index, inner_units_ratio=0.5):
        with tf.variable_scope("cbam_%s" % (index)):
            feature_map_shape = self.combined_static_and_dynamic_shape(feature_map)
            # channel attention
            channel_avg_weights = tf.nn.avg_pool(
                value=feature_map,
                ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            channel_max_weights = tf.nn.max_pool(
                value=feature_map,
                ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
                strides=[1, 1, 1, 1],
                padding='VALID'
            )
            channel_avg_reshape = tf.reshape(channel_avg_weights,
                                             [feature_map_shape[0], 1, feature_map_shape[3]])
            channel_max_reshape = tf.reshape(channel_max_weights,
                                             [feature_map_shape[0], 1, feature_map_shape[3]])
            channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)
            fc_1 = tf.layers.dense(
                inputs=channel_w_reshape,
                units=feature_map_shape[3] * inner_units_ratio,
                name="fc_1",
                activation=tf.nn.relu
            )
            fc_2 = tf.layers.dense(
                inputs=fc_1,
                units=feature_map_shape[3],
                name="fc_2",
                activation=None
            )
            channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
            channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
            channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
            feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)

            # spatial attention
            channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
            channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)
            channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                                  shape=[feature_map_shape[0], feature_map_shape[1],
                                                         feature_map_shape[2], 1])
            channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                                  shape=[feature_map_shape[0], feature_map_shape[1],
                                                         feature_map_shape[2], 1])
            channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
            spatial_attention = slim.conv2d(
                channel_wise_pooling,
                1,
                [1, 7],
                padding='SAME',
                activation_fn=tf.nn.sigmoid,
                scope="spatial_attention_conv"
            )
            feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
            return feature_map_with_attention
    def conv_layer(self, input_, filter_shape, bias_shape, dilation, causal, name, layer_id, padding, is_train=0):

        wName = name + "_W" + str(layer_id)
        bName = name + "_b" + str(layer_id)

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            W = tf.get_variable(wName, filter_shape,
                                initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
            b = tf.get_variable(bName, bias_shape,
                                initializer=tf.constant_initializer(0.01))
            if self.is_attention:
                input_ = self.conv_block_attention(input_, layer_id)
            if causal:
                p = (filter_shape[1] - 1) * dilation + 1 - input_.get_shape().as_list()[2]
                if p > 0:
                    pad_ = [[0, 0], [0, 0], [p, 0], [0, 0]]
                    input_ = tf.pad(input_, pad_)
                conv_out = tf.nn.atrous_conv2d(input_, W, rate=dilation, padding=padding) + b
            else:
                conv_out = tf.nn.conv2d(input_, W, strides=[1, 1, 1, 1], padding=padding) + b

            conv_out = self.layer_norm(conv_out, name="layer_norm", is_train=is_train)
            conv_out = tf.nn.relu(conv_out)
        return conv_out

    def Generator(self, X, user_seq, is_train):
        with tf.variable_scope('gen'):
            self.Neg_dim = tf.placeholder(tf.float32, [None, self.n_items])
            context_seq = X
            emb_items = tf.concat([self.item_embs, self.item_zero], axis=0)
            context_embedding = tf.nn.embedding_lookup(emb_items, context_seq, name="context_embedding")
            self.i_emb = context_embedding

            user_embedding = tf.nn.embedding_lookup(self.user_embs, user_seq, name="context_embedding")
            user_embedding = tf.reshape(user_embedding, [-1, self.emb_len])

            conv_hori_input = context_embedding
            conv_hori_out = tf.reshape(conv_hori_input, [-1, 1, int(self.L*self.percentage), self.emb_len])

            for layer_id, dilation in enumerate(self.dilations):
                conv_hori_out = self.conv_layer(conv_hori_out, [1, self.kernel_size[layer_id], self.emb_len, self.emb_len], [self.emb_len],
                                                dilation, True, 'conv_hori', layer_id, 'VALID', is_train)
            # self.i_emb = conv_hori_out
            conv_hori_out = tf.reshape(conv_hori_out, [-1, self.emb_len])

            # tf.print(conv_hori_input)
            conv_vert_input = context_embedding
            conv_vert_input = tf.reshape(conv_vert_input, [-1, int(self.L*self.percentage), self.emb_len, 1])
            conv_vert_out = self.conv_layer(conv_vert_input, [int(self.L*self.percentage), 1, 1, 1], [1], 1, False, 'conv_vert', 0, 'VALID', is_train)
            # self.u_emb = conv_vert_out
            conv_vert_out = tf.reshape(conv_vert_out, [-1, self.emb_len])

            # conv_vert_out = conv_vert_out*0
            # conv_hori_out = conv_hori_out*0
            fc_input = tf.concat([conv_hori_out, conv_vert_out, user_embedding], 1)
            fc_input_size = self.emb_len * 3

            logits, L2norm, W, b= self.fc_layer(fc_input, fc_input_size, self.n_items, 'none', 'gen_fc', 0, is_train)

            Neg_loss = tf.reduce_mean(tf.reduce_sum(tf.square(logits - 0) * self.Neg_dim, 1, keep_dims=True))

        return logits, L2norm, Neg_loss

    def Discriminator(self, input, h, activation, h_layers, is_train):

        # input->hidden
        y, _, W, b = self.fc_layer(input, self.n_items, h, activation, "dis", 0, is_train)

        # stacked hidden layers
        for layer in range(h_layers - 1):
            y, _, W, b = self.fc_layer(y, h, h, activation, "dis", layer + 1, is_train)

        # hidden -> output
        y, _, W, b = self.fc_layer(y, h, h, "none", "dis", h_layers, is_train)

        return y

    def model(self):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.user_seq = tf.placeholder(tf.int32, [None, 1])
        self.mask = tf.placeholder(tf.float32, [None, self.n_items])
        self.real_data = tf.placeholder(tf.float32, [None, self.n_items])
        self.is_train = tf.placeholder(tf.int16)

        self.G_output, G_L2norm, Neg_loss = self.Generator(self.X, self.user_seq, self.is_train)

        fakeData = self.G_output * self.mask

        D_real = self.Discriminator(self.real_data, self.model_args.hdim_D, 'tanh', self.model_args.hlayer_D, self.is_train)
        D_fake = self.Discriminator(fakeData, self.model_args.hdim_D, 'tanh', self.model_args.hlayer_D, self.is_train)

        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='dis')
        D_L2norm = 0
        for pr in d_vars:
            D_L2norm = D_L2norm + tf.nn.l2_loss(pr)

        if self.loss_type == 'wgan-gp':
            eps = tf.random_uniform([self.model_args.batchSize_D, 1], minval=0., maxval=1.)  # eps是U[0,1]的随机数
            X_inter = eps * self.real_data + (1. - eps) * fakeData  # 在真实样本和生成样本之间随机插值，希望这个约束可以“布满”真实样本和生成样本之间的空间
            grad = tf.gradients(self.Discriminator(X_inter, self.model_args.hdim_D, 'tanh', self.model_args.hlayer_D, self.is_train),
                                [X_inter])[0]  # 求梯度
            grad_norm = tf.sqrt(tf.reduce_sum((grad) ** 2, axis=1))  # 求梯度的二范数
            grad_pen = 10 * tf.reduce_mean(tf.nn.relu(grad_norm - 1.))

            # define loss for G
            g_loss = tf.reduce_mean(D_fake)
            self.g_loss = g_loss
            # define loss for D
            d_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake) + grad_pen
            self.d_loss = d_loss
        elif self.loss_type == 'gan':
            # define loss for G
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
            self.g_loss = g_loss + Neg_loss * 0.03
            # define loss for D
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
            self.d_loss = d_loss_fake + d_loss_real

        # define Optimizer
        if self.model_args.opt_D == 'sgd':
            self.trainer_G = tf.train.GradientDescentOptimizer(self.model_args.lr_G).minimize(self.g_loss, var_list=g_vars)
            self.trainer_D = tf.train.GradientDescentOptimizer(self.model_args.lr_D).minimize(self.d_loss, var_list=d_vars)
        elif self.model_args.opt_D == 'adam':
            self.trainer_G = tf.train.AdamOptimizer(self.model_args.lr_G).minimize(self.g_loss, var_list=g_vars)
            self.trainer_D = tf.train.AdamOptimizer(self.model_args.lr_D).minimize(self.d_loss, var_list=d_vars)
        elif self.model_args.opt_G == 'RMSProp':
            self.trainer_G = tf.train.RMSPropOptimizer(self.model_args.lr_G).minimize(self.g_loss, var_list=g_vars)
            self.trainer_D = tf.train.RMSPropOptimizer(self.model_args.lr_D).minimize(self.d_loss, var_list=d_vars)

        # only train G
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.real_data, logits=fakeData))
        if self.model_args.opt_G == 'sgd':
            self.trainer = tf.train.GradientDescentOptimizer(self.model_args.lr_G).minimize(self.loss)
        elif self.model_args.opt_G == 'adam':
            self.trainer = tf.train.AdamOptimizer(self.model_args.lr_G).minimize(self.loss)
        elif self.model_args.opt_G == 'RMSProp':
            self.trainer = tf.train.RMSPropOptimizer(self.model_args.lr_G).minimize(self.loss)