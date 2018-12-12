import numpy as np
import scipy.linalg
import tensorflow as tf

# fit (affine) warp between two sets of points 
def fit(Xsrc,Xdst):
	ptsN = len(Xsrc)
	X,Y,U,V,O,I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
	A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1),
						np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
	b = np.concatenate((U,V),axis=0)
	p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
	pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=np.float32)
	return pMtrx

# compute composition of warp parameters
def compose(config,p,dp):
	return p+dp

# compute composition of warp parameters
def inverse(config,p):
	return -p

# convert warp parameters to matrix
def vec2mtrx(config,p):
	with tf.name_scope("vec2mtrx"):
		if config.warpType=="homography":
			p1,p2,p3,p4,p5,p6,p7,p8 = tf.unstack(p,axis=1)
			A = tf.transpose(tf.stack([[p3,p2,p1],[p6,-p3-p7,p5],[p4,p8,p7]]),perm=[2,0,1])
		elif config.warpType=="affine":
			O = tf.zeros([config.batch_size])
			p1,p2,p3,p4,p5,p6 = tf.unstack(p,axis=1)
			A = tf.transpose(tf.stack([[p1,p2,p3],[p4,p5,p6],[O,O,O]]),perm=[2,0,1])
		else: assert(False)
		# matrix exponential
		pMtrx = tf.tile(tf.expand_dims(tf.eye(3),axis=0),[config.batch_size,1,1])
		numer = tf.tile(tf.expand_dims(tf.eye(3),axis=0),[config.batch_size,1,1])
		denom = 1.0
		for i in range(1,config.warpApprox):
			numer = tf.matmul(numer,A)
			denom *= i
			pMtrx += numer/denom
	return pMtrx

# warp the image
def transformImage(config,image,pMtrx):
	with tf.name_scope("transformImage"):
		refMtrx = tf.tile(tf.expand_dims(config.refMtrx,axis=0),[config.batch_size,1,1])
		transMtrx = tf.matmul(refMtrx,pMtrx)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,config.width),np.linspace(-1,1,config.height))
		X,Y = X.flatten(),Y.flatten()
		XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
		XYhom = np.tile(XYhom,[config.batch_size,1,1]).astype(np.float32)
		XYwarpHom = tf.matmul(transMtrx,XYhom)
		XwarpHom,YwarpHom,ZwarpHom = tf.unstack(XYwarpHom,axis=1)
		Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8),[config.batch_size,config.height,config.width])
		Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8),[config.batch_size,config.height,config.width])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		imageIdx = np.tile(np.arange(config.batch_size).reshape([config.batch_size,1,1]),[1,config.height,config.width])
		imageVec = tf.reshape(image,[-1,3])
		imageVecOut = tf.concat([imageVec,tf.zeros([1,3])],axis=0)
		idxUL = (imageIdx*config.height+YfloorInt)*config.width+XfloorInt
		idxUR = (imageIdx*config.height+YfloorInt)*config.width+XceilInt
		idxBL = (imageIdx*config.height+YceilInt)*config.width+XfloorInt
		idxBR = (imageIdx*config.height+YceilInt)*config.width+XceilInt
		idxOutside = tf.fill([config.batch_size,config.height,config.width],config.batch_size*config.height*config.width)
		def insideIm(Xint,Yint):
			return (Xint>=0)&(Xint<config.width)&(Yint>=0)&(Yint<config.height)
		idxUL = tf.where(insideIm(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideIm(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideIm(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideIm(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[config.batch_size,config.height,config.width,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[config.batch_size,config.height,config.width,1])
		imageUL = tf.to_float(tf.gather(imageVecOut,idxUL))*(1-Xratio)*(1-Yratio)
		imageUR = tf.to_float(tf.gather(imageVecOut,idxUR))*(Xratio)*(1-Yratio)
		imageBL = tf.to_float(tf.gather(imageVecOut,idxBL))*(1-Xratio)*(Yratio)
		imageBR = tf.to_float(tf.gather(imageVecOut,idxBR))*(Xratio)*(Yratio)
		imageWarp = imageUL+imageUR+imageBL+imageBR
	return imageWarp

# warp the image
def transformCropImage(config,image,pMtrx):
	with tf.name_scope("transformImage"):
		refMtrx = tf.tile(tf.expand_dims(config.refMtrx_b,axis=0),[config.batch_size,1,1])
		transMtrx = tf.matmul(refMtrx,pMtrx)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,config.width),np.linspace(-1,1,config.height))
		X,Y = X.flatten(),Y.flatten()
		XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
		XYhom = np.tile(XYhom,[config.batch_size,1,1]).astype(np.float32)
		XYwarpHom = tf.matmul(transMtrx,XYhom)
		XwarpHom,YwarpHom,ZwarpHom = tf.unstack(XYwarpHom,axis=1)
		Xwarp = tf.reshape(XwarpHom/(ZwarpHom+1e-8),[config.batch_size,config.height,config.W])
		Ywarp = tf.reshape(YwarpHom/(ZwarpHom+1e-8),[config.batch_size,config.height,config.W])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		imageIdx = np.tile(np.arange(config.batch_size).reshape([config.batch_size,1,1]),[1,config.height,config.W])
		imageVec = tf.reshape(image,[-1,3])
		imageVecOut = tf.concat([imageVec,tf.zeros([1,3])],axis=0)
		idxUL = (imageIdx*config.dataH+YfloorInt)*config.dataW+XfloorInt
		idxUR = (imageIdx*config.dataH+YfloorInt)*config.dataW+XceilInt
		idxBL = (imageIdx*config.dataH+YceilInt)*config.dataW+XfloorInt
		idxBR = (imageIdx*config.dataH+YceilInt)*config.dataW+XceilInt
		idxOutside = tf.fill([config.batch_size,config.height,config.W],config.batch_size*config.dataH*config.dataW)
		def insideIm(Xint,Yint):
			return (Xint>=0)&(Xint<config.dataW)&(Yint>=0)&(Yint<config.dataH)
		idxUL = tf.where(insideIm(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideIm(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideIm(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideIm(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[config.batch_size,config.height,config.W,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[config.batch_size,config.height,config.W,1])
		imageUL = tf.to_float(tf.gather(imageVecOut,idxUL))*(1-Xratio)*(1-Yratio)
		imageUR = tf.to_float(tf.gather(imageVecOut,idxUR))*(Xratio)*(1-Yratio)
		imageBL = tf.to_float(tf.gather(imageVecOut,idxBL))*(1-Xratio)*(Yratio)
		imageBR = tf.to_float(tf.gather(imageVecOut,idxBR))*(Xratio)*(Yratio)
		imageWarp = imageUL+imageUR+imageBL+imageBR
	return imageWarp
