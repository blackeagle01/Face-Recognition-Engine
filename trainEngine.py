from keras.applications import vgg16 
from keras import backend as K
import tensorflow as tf 
from keras.models import Model


# Setting the learning phase to 1 indicating we want to train the model.
K.set_learning_phase(1)


# Using the tensorflow interface to keras
sess=tf.Session()
K.set_session(sess)


# Getting a 4096 length encoding of the input image  
vgg=vgg16.VGG16(include_top=True)
layer=vgg.get_layer(name='fc2')
output=layer.output


# Creating a custom model which takes input the image and returns a 4096 length encoding
model=Model(inputs=vgg.input,outputs=[output])

A=K.placeholder(shape=(224,224,3),dtype=K.floatx()) # Placeholder for the anchor image
P=K.placeholder(shape=(224,224,3),dtype=K.floatx())	# Placeholder for the postive image
N=K.placeholder(shape=(224,224,3),dtype=K.floatx()) # Placeholder for the negative image

alpha=0.5 # margin


# siamese eqn= norm(f(A)-f(P))-norm(f(A)-f(N))+alpha
siamese_eqn=tf.losses.mean_squared_error(labels=model(A), predictions=model(P))-tf.losses.mean_squared_error(labels=model(A), predictions=model(N))+alpha



# loss= max(siamese_eqn,0)

loss=tf.maximum(x=siamese_eqn, y=K.zeros_like(siamese_eqn))


train=tf.train.AdamOptimizer(0.001).minimize(loss)

sess.run(tf.global_variables_initializer())
feed_dict=load_feed_dict()


#train the network
for i in range(n_iterations):
	sess.run(train,feed_dict=feed_dict)

print(sess.run(loss,feed_dict=feed_dict))

model.save('deepface.h5')


