# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:07:53 2022

@author: Noel
"""

import os
import skimage.io
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.disable_eager_execution()

#x1 = tf.constant([1,2,3,4,5])
#x2 = tf.constant([6,7,8,9,10])

#with tf.compat.v1.Session() as tfSes:
#    res = tfSes.run(x1*x2)

#print(res)

#config = tf.ConfigProto(log_device_placement = True)
#config = tf.ConfigProto(allow_soft_placement = True)


#APRENDIZAJE NEURONAL DE LAS SEÑALES DE TRAFICO
#----------------------------------------------

def load_ml_data(data_directory):
    dirs = [d for d in os.listdir(data_directory)
            if os.path.isdir(os.path.join(data_directory,d))]
    
    labels = []
    images = []
    
    for d in dirs:
        label_dir = os.path.join(data_directory, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".ppm")]
        
        for f in file_names:
            images.append(skimage.io.imread(f))
            labels.append(int(d))
            
    return images, labels
        


main_dir = "../datasets/BelgiumTS/"
train_data_dir = os.path.join(main_dir, "Training")
test_data_dir = os.path.join(main_dir, "Testing")


images, labels = load_ml_data(train_data_dir)

images = np.array(images, list)
labels = np.array(labels, list)


plt.hist(labels, len(set(labels)))

rand_signs = random.sample(range(0, len(labels)), 6)

print(rand_signs)

for i in range(len(rand_signs)):
    temp_im = images[rand_signs[i]]
    plt.subplot(1, 6, i+1)
    plt.axis("off")
    plt.imshow(temp_im)
    plt.subplots_adjust(wspace= 0.5)
    plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(temp_im.shape,
                                               temp_im.min(),
                                               temp_im.max()))
    
unique_labels = set(labels)
plt.figure(figsize=(16,16))
i = 1
for label in unique_labels:
    temp_im = images[list(labels).index(label)]
    plt.subplot(8, 8, i)
    plt.axis("off")
    plt.title("Clase {0} ({1})".format(label, list(labels).count(label)))
    i += 1
    plt.imshow(temp_im)
plt.show()

# MODELO DE RED NEURONAL CON TENSORFLOW
# No todas las images son del mismo tamaño
# Hay 62 clases de imagenes
# La distribución de señales de trafico no es uniforme(algunas salen mas veces que otras)
              
w = 9999
h = 9999

for image in images:
    if image.shape[0] < h:
        h = image.shape[0]
    if image.shape[1] < w:
        w = image.shape[1]
        
print("Tamaño minimo: {0}x{1}".format(h,w))

images30 = [transform.resize(image, (30, 30)) for image in images]

for i in range(len(rand_signs)):
    temp_im = images30[rand_signs[i]]
    plt.subplot(1, 6, i+1)
    plt.axis("off")
    plt.imshow(temp_im)
    plt.subplots_adjust(wspace= 0.5)
    plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(temp_im.shape,
                                               temp_im.min(),
                                               temp_im.max()))
    

images30 = np.array(images30)
images30 = rgb2gray(images30)

for i in range(len(rand_signs)):
    temp_im = images30[rand_signs[i]]
    plt.subplot(1, 6, i+1)
    plt.axis("off")
    plt.imshow(temp_im, cmap="gray")
    plt.subplots_adjust(wspace= 0.5)
    plt.show()
    print("Forma:{0}, min:{1}, max:{2}".format(temp_im.shape,
                                               temp_im.min(),
                                               temp_im.max()))
    

# USANDO TENSOR FLOW (TAMBIEN EXISTE KERAX)

# D:\CursosExternos\ML-DataScience\mis-notebooks\T1-1-TensorFlow.py:148: UserWarning: `tf.layers.flatten` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Flatten` instead.
#  images_flat = tfv1.compat.v1.layers.flatten(x)
#D:\CursosExternos\ML-DataScience\mis-notebooks\T1-1-TensorFlow.py:149: UserWarning: `tf.layers.dense` is deprecated and will be removed in a future version. Please use `tf.keras.layers.Dense` instead.
#  logits = tfv1.compat.v1.layers.dense(images_flat, 62, tf.nn.relu)

import tensorflow.compat.v1 as tfv1
tfv1.disable_v2_behavior()

x = tfv1.compat.v1.placeholder(dtype = tf.float32, shape = [None, 30, 30])
y = tfv1.compat.v1.placeholder(dtype = tf.int32, shape = [None])

# ---- construyendo la RED NEURONAL ----
images_flat = tfv1.compat.v1.layers.flatten(x)
logits = tfv1.compat.v1.layers.dense(images_flat, 62, tf.nn.relu)
    
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

# funcion para optimizar (metodo de Adam, pero hay mas)
train_opt = tfv1.compat.v1.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

final_pred = tf.argmax(logits, 1)
    
accuracy = tf.reduce_mean(tf.cast(final_pred, tf.float32))


# Para la ejecución de la red neuronal
tfv1.compat.v1.random.set_random_seed(1234)

sess = tf.compat.v1.Session()

sess.run(tfv1.compat.v1.global_variables_initializer())
    
# 300 es el entranimiento de la red neuronal( puedo poner mas )
for i in range(3000):

    _, accuracy_val = sess.run([train_opt, accuracy],
                               feed_dict = {
                                   x: images30,
                                   y: list(labels)
                                })
    
    _, loss_val = sess.run([train_opt, loss],
                               feed_dict = {
                                   x: images30,
                                   y: list(labels)
                                })
    
    if i%10 == 0:
        print("EPOCH", i)
        print("Eficacia: ", accuracy_val)
        print("Perdidas", loss_val)
    #print("Fin del Ecpoh ", i)
    

# EVALUACION DE LA RED NEURONAL

sample_idx = random.sample(range(len(images30)), 40)
sample_images = [images30[i] for i in sample_idx]
sample_labels = [labels[i] for i in sample_idx]

prediction = sess.run([final_pred], feed_dict={x:sample_images})[0]

plt.figure(figsize=(16,20))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    predi = prediction[i]
    plt.subplot(10,4,i+1)
    plt.axis("off")
    color = "green" if truth==predi else "red"
    plt.text(32,15, "Real:       {0}\nPrediction:{1}".format(truth, predi),
             fontsize = 14, color = color)
    plt.imshow(sample_images[i], cmap="gray")
plt.show()




