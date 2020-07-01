# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:05:59 2019

@author: liy0o
"""

import numpy as np
import time
import scipy.io
import h5py
import keras.backend as K
import tensorflow as tf
from numpy import random
import math
from keras import losses 
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Subtract, Multiply, Flatten, Average, merge, Dot, DepthwiseConv2D, Conv2D, ConvLSTM2D, LocallyConnected2D, Conv3D,Activation, BatchNormalization, AveragePooling2D, Add, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate, concatenate, Lambda, Reshape, Conv2DTranspose, Flatten, Dense
from activations import binary_sigmoid as binary_sigmoid_op
from binary_layers import BinaryDense, BinaryConv2D
from keras.layers import GaussianNoise
from keras import regularizers
from keras.constraints import NonNeg, UnitNorm
from masklayer import MaskLayer




num_iter = 19
num_filters = 64


class ElapsedTimer(object):
	def __init__(self):
		self.start_time = time.time()
	def elapsed(self,sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"
	def elapsed_time(self):
		print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )


class OPTSAM():
	def __init__(self, img_rows = 256, img_cols = 256, img_ranks = 32):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.img_ranks = img_ranks
	
	
	def residual_block(self, layer_input, filters, num_channels = 32):

		d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
		d = Activation('relu')(d)
		d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
		d = Activation('relu')(d)
		d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
		d = Activation('relu')(d)
		d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
		d = Activation('relu')(d)

		d = Conv2D(num_channels, kernel_size=3, strides=1, padding='same')(d)

		d = Add()([d, layer_input])
		return d

		
	def get_convnet(self): 
		
		# INPUTS
		inputs = Input((256,256,32))

		gt = inputs
		mylayer = MaskLayer((256,256,8))
		biasvalue = mylayer(Lambda(lambda x:x[:,:,:,0:8])(gt))

		phi = Activation(binary_sigmoid_op, name='phi')(biasvalue)
		phi = Reshape((256,256,8))(phi)
		
		
		def TimesPhi(input_img, phi):  # 32->4

			addedlist = []
			for i in range(4):
				temp_img = Lambda(lambda x: x[:,:,:,8*i:8*i+8])(input_img)
				temp_img = Multiply()([temp_img, phi])				
				tempadded = Lambda(lambda x: K.sum(x, axis=-1))(temp_img)

				addedlist.append(Lambda(lambda x: K.expand_dims(x))(tempadded))				
#			tempadded = Lambda(lambda x: K.squeeze(x,-1))(tempadded)
			v_added = Concatenate(axis=-1)(addedlist)

			return v_added 
			
			
			
		def TimesTransponsePhi(input_img, phi): # 4->32
#			print("input trans phi:",input_img.shape)
			addedlist = []
#			print(input_img.shape)
			for i in range(4):
				temp_img = Lambda(lambda x: K.expand_dims(x[:,:,:,i]))(input_img)
				expand_img = Lambda(K.tile,arguments={'n':(1,1,1,8)})(temp_img)
				tempadded =  Multiply()([expand_img, phi])
				addedlist.append(tempadded)
			return Concatenate(axis=-1)(addedlist)

		
		def TimesPhiPhi(input_img, phi):
			temp_img = TimesPhi(input_img, phi)
			temp_img = TimesTransponsePhi(temp_img, phi)
			return temp_img
		
		
		
		def denoiseblock(x):
			temp = x
			temp = self.residual_block(temp, num_filters)
			temp = self.residual_block(temp, num_filters)
			return temp 
			
		

		def faststage(prex, lastx, v, Phiy, phi):
			beta = DepthwiseConv2D(1,use_bias= False,depthwise_constraint=NonNeg())(Subtract()([lastx,prex]))

			x = Add()([lastx,beta])

			diff_yx = Subtract()([Phiy,TimesPhiPhi(x, phi)])
			diff_vx = Subtract()([v,x])

			diff_yx = DepthwiseConv2D(1,use_bias= False,depthwise_constraint=NonNeg())(diff_yx)
			diff_vx = DepthwiseConv2D(1,use_bias= False,depthwise_constraint=NonNeg())(diff_vx)

			x_next = Add()([x,diff_yx,diff_vx,beta])
			v_next = denoiseblock(x_next)

			return x_next, v_next


		y = TimesPhi(gt, phi)
#		y = GaussianNoise(0.1)(y)
		Phiy = TimesTransponsePhi(y, phi)
		
		x0 = Phiy
		v0 = x0

		tempx, tempv = faststage(x0, x0, v0, Phiy, phi)
		lastx = tempx
		prex = x0
		
		for i in range(num_iter-2):
			tempx, tempv = faststage(prex, lastx, tempv, Phiy,phi)
			prex = lastx
			lastx = tempx
			
		resx, resv = faststage(prex, lastx,tempv,Phiy,phi)

		resv = Activation('sigmoid')(resv) 
		

		def PSNR(y_true, y_pred):
			return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
		
		def SSIM(y_true, y_pred):
			return tf.image.ssim(y_true, y_pred, 1)
		# Setup the model inputs / outputs
		model = Model(inputs=inputs, outputs=resv)

		model.compile(
			optimizer = Adam(lr=0.001),
			loss= 'mse',
			metrics=[PSNR,SSIM]
		)
		model.summary()
		
		maskmodel = Model(inputs=inputs, outputs=phi)
		
		return model, maskmodel	


	
	def train(self):

		print("loading data")
		imgs_train,  imgs_test = self.load_data()
		print("loading data done")

		model,maskmodel = self.get_convnet()
#		model.load_weights('opt_video_rec.hdf5')
		model_checkpoint = ModelCheckpoint('opt_video_rec.hdf5', monitor='loss',verbose=1, save_best_only=True)

		history = model.fit(imgs_train, imgs_train, batch_size=1, epochs= 300 ,verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		imgs_rec_test = model.predict(imgs_test, batch_size=1, verbose=1)
		opt_mask = maskmodel.predict(imgs_train, batch_size=1, verbose=1)



		

if __name__ == '__main__':
	optmask_net = OPTSAM()
	timer = ElapsedTimer()
	optmask_net.train()
	timer.elapsed_time()





