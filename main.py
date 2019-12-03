'''import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import matplotlib.pyplot as plt
#用来正常显示中文
plt.rcParams["font.sans-serif"]=["SimHei"]
 
if __name__ == "__main__":
    img = cv2.imread("image2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    crop_img = tf.random_crop(img,[560,560,3])
    std_img = tf.image.per_image_standardization(crop_img)

    sess = tf.InteractiveSession()
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img)
    plt.title("原始图片")
    plt.subplot(122)
    #crop_img = crop_img.eval()
    crop_img = std_img.eval()
    plt.title("裁剪后的图片")
    plt.imshow(crop_img)
    plt.show()

exit()'''

'''pc = 0.4 + random.random() * 0.2
mi_x = 0.4 + random.random() * 0.2
mi_y = 0.4 + random.random() * 0.2
x0 = int(S * mi_x - S * pc * 0.5)
x1 = int(S * mi_x + S * pc * 0.5)
y0 = int(S * mi_y - S * pc * 0.5)
y1 = int(S * mi_y + S * pc * 0.5)
cut_image = image[x0 : x1, y0 : y1]

cv2.imshow('fig', cut_image)
cv2.waitKey(1)

feature = np.reshape(cv2.resize(cut_image, (SIZE, SIZE)), SIZE * SIZE)'''

import os
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random

ORI_SIZE = 640
SIZE = 32
CHANNEL = 3
EPOCH = 5000
BATCH_SIZE = 50

# *************** 构建多层卷积网络 *************** #

batch = tf.placeholder("int32")
x = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE, CHANNEL])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder("float")

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([3, 3, CHANNEL, 32])
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])
W_fc1 = weight_variable([SIZE * SIZE // 64 * 128, 1024])
b_fc1 = bias_variable([1024])

h_conv1 = tf.nn.sigmoid(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_conv3 = tf.nn.sigmoid(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool3_flat = tf.reshape(h_pool3, [-1, SIZE * SIZE // 64 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv=tf.nn.softmax(logits)

# *************** 训练和评估模型 *************** #

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
loss = tf.reduce_mean(cross_entropy)
saver = tf.train.Saver()

# *************** 开始训练模型 *************** #

def random_cut(image):
    x0 = int(ORI_SIZE * (0.1 + random.random() * 0.3))
    x1 = int(ORI_SIZE * (0.6 + random.random() * 0.3))
    y0 = int(ORI_SIZE * (0.1 + random.random() * 0.3))
    y1 = int(ORI_SIZE * (0.6 + random.random() * 0.3))
    cut_image = cv2.resize(image[x0 : x1, y0 : y1], (SIZE, SIZE))
    return cut_image

root = './Eyes/'
files = os.listdir(root)

X = []
Y = []
X_test = []
Y_test = []

cnt = 0
for file in files:
    tags = file.split('.')[0].split('_')
    user = int(tags[0])
    dist = int(tags[1][:-1])
    P = int(tags[2][:-1])
    V = int(tags[3][:-1])
    H = int(tags[4][:-1])
    if (V == 0 and H == 0):
        label = 1
        truth = [0, 1]
    else:
        label = 0
        truth = [1, 0]

    image = cv2.imread(root + file)
    image = image[0 : ORI_SIZE, ORI_SIZE * 1: ORI_SIZE * 2]
    cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=1)
    #cv2.imshow('fig', image)
    #cv2.waitKey(1)

    for r in range(1 + 14 * label):
        if user % 5 != 0:
            X.append(image)
            Y.append(truth)
        else:
            X_test.append(image)
            Y_test.append(truth)

    cnt += 1
    if cnt % 100 == 0:
        print('Input', cnt)
    #if cnt == 1000:
    #    break

N = len(X)
N_test = len(X_test)
state = np.random.get_state()
np.random.set_state(state)
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(Y)

#Train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCH):
        X_image = []
        X_test_image = []
        for i in range(N):
            X_image.append(random_cut(X[i]))
        for i in range(N_test):
            X_test_image.append(random_cut(X_test[i]))

        for i in range(0, N - BATCH_SIZE, BATCH_SIZE):
            start = i
            end = min(i + BATCH_SIZE, N)
            batch_X = X_image[start: end]
            batch_Y = Y[start: end]
            train_step.run(feed_dict={x: batch_X, y_: batch_Y, keep_prob: 0.5, batch: end - start})
        (train_loss, train_acc) = sess.run((loss, accuracy), feed_dict={x:X_image[:N_test], y_: Y[:N_test], keep_prob: 1.0, batch: N_test})
        (test_loss, test_acc) = sess.run((loss, accuracy), feed_dict={x: X_test_image, y_: Y_test, keep_prob: 1.0, batch: N_test})
        print('{:<20}{:<20}{:<20}{:<20}{:<20}'.format(epoch,train_loss,test_loss,train_acc,test_acc))

    saver.save(sess, './save/model.ckpt')
