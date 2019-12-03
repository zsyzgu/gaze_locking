import tensorflow.compat.v1 as tf
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

exit()

import os
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random

SIZE = 32 
classes = 2
CHANNEL = 1

# *************** 构建多层卷积网络 *************** #

x = tf.placeholder(tf.float32, shape=[None, SIZE * SIZE * CHANNEL])
y_ = tf.placeholder(tf.float32, shape=[None, classes])
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

x_image = tf.reshape(x, [-1,SIZE,SIZE,CHANNEL])

W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
W_conv3 = weight_variable([3, 3, 64, 128])
b_conv3 = bias_variable([128])

W_fc1 = weight_variable([SIZE * SIZE // 64 * 128, 1024])
b_fc1 = bias_variable([1024])

h_conv1 = tf.nn.sigmoid(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_conv3 = tf.nn.sigmoid(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool4_flat = tf.reshape(h_pool3, [-1, SIZE * SIZE // 64 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, classes])
b_fc2 = bias_variable([classes])

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
  else:
    label = 0

  image = cv2.imread(root + file, cv2.IMREAD_GRAYSCALE)
  S = np.size(image, 0)
  image = image[0 : S, S * 1: S * 2]

  REP = 5
  if label == 0:
    rep = REP
  else:
    rep = REP * 14

  for r in range(rep):
    pc = 0.4 + random.random() * 0.2
    mi_x = 0.4 + random.random() * 0.2
    mi_y = 0.4 + random.random() * 0.2
    x0 = int(S * mi_x - S * pc * 0.5)
    x1 = int(S * mi_x + S * pc * 0.5)
    y0 = int(S * mi_y - S * pc * 0.5)
    y1 = int(S * mi_y + S * pc * 0.5)
    cut_image = image[x0 : x1, y0 : y1]

    #cv2.imshow('fig', cut_image)
    #cv2.waitKey(1)

    feature = np.reshape(cv2.resize(cut_image, (SIZE, SIZE)), SIZE * SIZE)

    truth = [0] * classes
    truth[label] = 1
    if user % 5 != 0:
      X.append(feature)
      Y.append(truth)
    else:
      X_test.append(feature)
      Y_test.append(truth)

  cnt += 1
  print(cnt)
  #if cnt == 1000:
  #  break

X = np.reshape(X, (-1, SIZE * SIZE * CHANNEL))
Y = np.reshape(Y, (-1, classes))
X_test = np.reshape(X_test, (-1, SIZE * SIZE * CHANNEL))
Y_test = np.reshape(Y_test, (-1, classes))

N = len(X)
state = np.random.get_state()
np.random.set_state(state)
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(Y)

batch_size = 500

#Train
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for e in range(5000):
    for i in range(0, N - batch_size, batch_size):
      start = i
      end = min(i + batch_size, N)
      batch_X = X[start: end]
      batch_Y = Y[start: end]
      train_step.run(feed_dict={x: batch_X, y_: batch_Y, keep_prob: 0.5})
    
    train_loss = loss.eval(feed_dict={x:X[:len(X_test)], y_: Y[:len(X_test)], keep_prob: 1.0})
    test_loss = loss.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
    train_acc = accuracy.eval(feed_dict={x: X[:len(X_test)], y_: Y[:len(X_test)], keep_prob: 1.0})
    test_acc = accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0})
    print(e, train_loss, test_loss, train_acc, test_acc)

  saver.save(sess, './save/model.ckpt')


#Test
'''
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  ckpt = tf.train.latest_checkpoint('./save/')
  saver.restore(sess, ckpt)
  print("test accuracy %g"%accuracy.eval(feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0}))
'''
