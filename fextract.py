#!/usr/bin/env python
# coding: utf-8

# In[37]:


from sklearn.utils import shuffle
from time import ctime
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[38]:


def clipread(paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB', interp='bilinear'):
    """
    Read video clip, resize to height and width resolution, crop the clip, then resize to crop height and crop width
    :param paths: Paths to  N (N = 16 for C3D) consecutive frames
    :param offsets: Crop window offset in form of [from_H, to_H, from_W, to_W], example: (0, 112, 24, 136)
    :param size: Tuple, size of the output image
    :param crop_size: Tuple, size of the output cropped image
    :param mode: 'RGB' or 'L' for gray scale
    :param interp: Interpolation to use for re-sizing, example: 'nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic'
    :return: Cropped clip (depth, crop_height, crop_width, channels) in float32 format, pixel values in [0, 255]
    """
    assert mode in ('RGB', 'L'), 'Mode is either RGB or L'

    clips = []
    for file_name in paths:
        # Read video frame
        im = imread(file_name, mode=mode)

        # Resize frame to init resolution and crop then resize to target resolution
        if mode == 'RGB':
            im = imresize(im, size=size, interp=interp)
            data = im[offsets[0]:offsets[1], offsets[2]:offsets[3], :]
            im = imresize(data, size=crop_size, interp=interp)
        else:
            im = imresize(im, size=size, interp=interp)
            data = im[offsets[0]:offsets[1], offsets[2]:offsets[3]]
            im = imresize(data, size=crop_size, interp=interp)

        clips.append(im)

    clips = np.array(clips, dtype=np.float32)

    if mode == 'RGB':
        return clips
    return np.expand_dims(clips, axis=3)


# In[39]:


def randcrop(scales, size=(128, 171)):
    """
    Generate random offset for crop window
    :param scales: List of scales for crop window, example: (128, 112, 96, 84)
    :param size: Tuple, size of the image
    :return: Crop window offsets in form of (from_H, to_H, from_W, to_W), example: (0, 112, 24, 136)
    """
    scales = np.array(scales) if isinstance(scales, (list, tuple)) else np.array([scales])
    scale = scales[np.random.randint(len(scales))]
    height, width = size

    max_h = height - scale
    max_w = width - scale

    off_h = np.random.randint(max_h) if max_h > 0 else 0
    off_w = np.random.randint(max_w) if max_w > 0 else 0

    return off_h, off_h + scale, off_w, off_w + scale


# In[40]:


HEIGHT = 128
WIDTH = 171
FRAMES = 16
CROP_SIZE = 112
CHANNELS = 3
BATCH_SIZE = 32


# In[41]:


def c3d_ucf101_finetune(inputs, training, weights=None):
    """
    C3D network for ucf101 dataset fine-tuned from weights pretrained on Sports1M
    :param inputs: Tensor inputs (batch, depth=16, height=112, width=112, channels=3), should be means subtracted
    :param training: A boolean tensor for training mode (True) or testing mode (False)
    :param weights: pretrained weights, if None, return network with random initialization
    :return: Output tensor for 101 classes
    """

    # create c3d network with pretrained Sports1M weights
    net = tf.layers.conv3d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[0]),
                           bias_initializer=tf.constant_initializer(weights[1]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[2]),
                           bias_initializer=tf.constant_initializer(weights[3]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[4]),
                           bias_initializer=tf.constant_initializer(weights[5]))
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[6]),
                           bias_initializer=tf.constant_initializer(weights[7]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[8]),
                           bias_initializer=tf.constant_initializer(weights[9]))
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[10]),
                           bias_initializer=tf.constant_initializer(weights[11]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[12]),
                           bias_initializer=tf.constant_initializer(weights[13]))
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[14]),
                           bias_initializer=tf.constant_initializer(weights[15]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[16]),
                          bias_initializer=tf.constant_initializer(weights[17]))
    net = tf.identity(net, name='fc1')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[18]),
                          bias_initializer=tf.constant_initializer(weights[19]))
    net = tf.identity(net, name='fc2')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=101, activation=None,
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
                          bias_initializer=tf.zeros_initializer())

    net = tf.identity(net, name='logits')

    return net


# In[42]:


def read_train(tr_file):
    path, frm, cls = tr_file.split(' ')
    start = np.random.randint(int(frm) - FRAMES)

    frame_dir = '.\frames\\'

    v_paths = [frame_dir + path + '*_v_*.jpg' % (f + 1) for f in range(start, start + FRAMES)]

    offsets = randcrop(scales=[128, 112, 96, 84], size=(128, 171))
    voxel = clipread(v_paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB')

    is_flip = np.random.rand(1, 1).squeeze() > 0.5
    if is_flip:
        voxel = np.flip(voxel, axis=2)

    return voxel, np.float32(cls)


# In[43]:


def read_test(tst_file):
    path, start, cls, vid = tst_file.split(' ')
    print(path, start, cls, vid)
    start = int(start)

    frame_dir = '.\frames\\'
    v_paths = [frame_dir + path + '*_v_*' + str(f) +'.jpg' for f in range(start, start + FRAMES)]
    print (v_paths)
    offsets = [8, 8 + 112, 30, 30 + 112] # center crop
    voxel = clipread(v_paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB')

    return voxel, np.float32(cls), np.float32(vid)


# In[44]:


def demo_finetune():
    # Demo of training on UCF101
    with open('.\list\c3d_train01.txt', 'r') as f:
        lines = f.read().split('\n')
    tr_files = [line for line in lines if len(line) > 0]

    weights = sio.loadmat('pretrained\c3d_sports1m_tf.mat', squeeze_me=True)['weights']

    # Define placeholders
    x = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
    y = tf.placeholder(tf.int64, None, name='input_y')
    training = tf.placeholder(tf.bool, name='training')

    # Define the C3D model for UCF101
    inputs = x - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
    logits = c3d_ucf101_finetune(inputs=inputs, training=training, weights=weights)
    labels = tf.one_hot(y, 101, name='labels')

    # Some operations
    correct_opt = tf.equal(tf.argmax(logits, 1), y, name='correct')
    acc_opt = tf.reduce_mean(tf.cast(correct_opt, tf.float32), name='accuracy')

    # Define training opt
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='loss')

    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=1000,
                                               decay_rate=0.96, staircase=True)
    train_opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        n_train = len(tr_files)
        for epoch in range(30):
            tr_files = shuffle(tr_files)
            batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
            batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)

            bidx = 0
            for idx, tr_file in enumerate(tr_files):
                voxel, cls = read_train(tr_file)

                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_train:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: True}
                    _, lss, acc = sess.run([train_opt, loss, acc_opt], feed_dict=feeds)

                    print ('%04d/%04d/%04d, loss: %.3f, acc: %.2f' % (idx / BATCH_SIZE, idx, n_train, lss, acc), ctime())

                    # reset batch
                    bidx = 0


# In[45]:


def c3d_ucf101(inputs, training, weights=None):
    """
    C3D network for ucf101 dataset, use pretrained weights on UCF101 for testing
    :param inputs: Tensor inputs (batch, depth=16, height=112, width=112, channels=3), should be means subtracted
    :param training: A boolean tensor for training mode (True) or testing mode (False)
    :param weights: pretrained weights, if None, return network with random initialization
    :return: Output tensor for 101 classes
    """

    # create c3d network with pretrained ucf101 weights
    net = tf.layers.conv3d(inputs=inputs, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[0]),
                           bias_initializer=tf.constant_initializer(weights[1]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[2]),
                           bias_initializer=tf.constant_initializer(weights[3]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[4]),
                           bias_initializer=tf.constant_initializer(weights[5]))
    net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[6]),
                           bias_initializer=tf.constant_initializer(weights[7]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[8]),
                           bias_initializer=tf.constant_initializer(weights[9]))
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.constant_initializer(weights[10]),
                           bias_initializer=tf.constant_initializer(weights[11]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])

    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[12]),
                           bias_initializer=tf.constant_initializer(weights[13]))
    net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
    net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                           kernel_initializer=tf.constant_initializer(weights[14]),
                           bias_initializer=tf.constant_initializer(weights[15]))
    net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[16]),
                          bias_initializer=tf.constant_initializer(weights[17]))
    net = tf.identity(net, name='fc1')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                          kernel_initializer=tf.constant_initializer(weights[18]),
                          bias_initializer=tf.constant_initializer(weights[19]))
    net = tf.identity(net, name='fc2')
    net = tf.layers.dropout(inputs=net, rate=0.5, training=training)

    net = tf.layers.dense(inputs=net, units=101, activation=None,
                          kernel_initializer=tf.constant_initializer(weights[20]),
                          bias_initializer=tf.constant_initializer(weights[21]))
    net = tf.identity(net, name='logits')

    return net


# In[46]:


def demo_test():
    # Demo of testing on UCF101
    # Each line in test/val file is in form: path start_frame class video
    with open('.\list\c3d_test01.txt', 'r') as f:
        lines = f.read().split('\n')
    tst_files = [line for line in lines if len(line) > 0]

    weights = sio.loadmat('pretrained\c3d_ucf101_tf.mat', squeeze_me=True)['weights']

    # Define placeholders
    x = tf.placeholder(tf.float32, shape=(None, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), name='input_x')
    y = tf.placeholder(tf.int64, None, name='input_y')
    training = tf.placeholder(tf.bool, name='training')

    # Define the C3D model for UCF101
    # inputs = x - tf.constant([90.2, 97.6,  101.4], dtype=tf.float32, shape=[1, 1, 1, 1, 3])
    inputs = x - tf.constant([96.6], dtype=tf.float32, shape=[1, 1, 1, 1, 1])
    logits = c3d_ucf101(inputs=inputs, training=training, weights=weights)

    correct_opt = tf.equal(tf.argmax(logits, 1), y, name='correct')
    acc_opt = tf.reduce_mean(tf.cast(correct_opt, tf.float32), name='accuracy')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        n_sample = len(tst_files)
        for epoch in range(1):

            batch_x = np.zeros(shape=(BATCH_SIZE, FRAMES, CROP_SIZE, CROP_SIZE, CHANNELS), dtype=np.float32)
            batch_y = np.zeros(shape=BATCH_SIZE, dtype=np.float32)
            
            accuracy = []
            scores = []
            
            clss = []
            vids = []
            
            bidx = 0
            for idx, tst_file in enumerate(tst_files):
                voxel, cls, vid = read_test(tst_file)
                batch_x[bidx] = voxel
                batch_y[bidx] = cls
                
                clss.append(cls)
                vids.append(vid)

                bidx += 1

                if (idx + 1) % BATCH_SIZE == 0 or (idx + 1) == n_sample:
                    feeds = {x: batch_x[:bidx], y: batch_y[:bidx], training: False}
                    acc, score = sess.run([acc_opt, logits], feed_dict=feeds)

                    scores.append(score)
                    accuracy.append(acc * bidx)

                    print ('%05d/%05d' % (idx, n_sample), 'acc: %.2f' % acc, ctime())

                    # reset batch
                    bidx = 0

            print ('Acc: ', np.sum(accuracy) / n_sample)
            mat = dict()
            mat['scores'] = np.vstack(scores).transpose()
            mat['cls'] = np.array(clss)
            mat['vid'] = np.array(vids)

            sio.savemat('.\c3d_demo_01.mat', mat)


# In[ ]:


if __name__ == '__main__':
    demo_test()
    # demo_finetune()


# In[ ]:




