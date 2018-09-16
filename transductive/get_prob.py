import os
f = open('prob_list.txt').readlines()
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math
import sys
runlist = f[int(sys.argv[1]):int(sys.argv[2])]
for index,line in enumerate(runlist):
	print(index)
	line = line.strip().split()[0]
	prob = np.load(line).astype(np.float32)
	#prob = torch.from_numpy(prob).cuda()
	#prob = torch.exp(prob)
	#prob = prob.cpu().numpy()
	prob = np.exp(prob)
	argmaxprob = np.argmax(prob, axis = 2)
	softprob = prob.max(2)
	cate_max = []
	for i in range(26):
		cate_max.append(prob[:,:,i].max())
	aa = set()
	for i in range(argmaxprob.shape[0]):
		for j in range(argmaxprob.shape[1]):
			max_cate = argmaxprob[i][j]
			if cate_max[max_cate] > 0.45 and softprob[i][j] > cate_max[max_cate] * 0.5 and softprob[i][j] < 0.5:
				aa.add(max_cate)
				softprob[i][j] = 1
	argmaxprob[softprob<0.5] = 255
	argmaxprob = np.clip(argmaxprob, 0, 255)
	cv2.imwrite('test_gt1/' + sys.argv[1] + '_%d.png' % index,argmaxprob)

