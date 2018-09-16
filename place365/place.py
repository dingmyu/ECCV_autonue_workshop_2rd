# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
from densenet import *
import numpy as np
# th architecture to use
arch = 'densenet161'

# load the pre-trained weights
model_file = 'densenet161_places365.pth.tar' 
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = densenet161(num_classes=365)
print(model)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
#model.classifier = F.softmax
model.eval().cuda()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

f = open('newtestlist.txt').readlines()
f2 = open('place_test.txt', 'w')
for index, line in enumerate(f):
    line1 = line.strip()
    line = line.strip().split()[0]
    img = Image.open(line)
    input_img = V(centre_crop(img).unsqueeze(0)).cuda()

# forward pass
    logit = model.forward(input_img)
    logit = logit.data.cpu().numpy()
    np.save('place_test/%d' % index, logit)
    f2.write(line + ' place_test/%d.npy\n' % index)
