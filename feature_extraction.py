
from networks.ArcFace import ArcFace
import torch.nn.functional as F

    
def get_feature(data):
	#print(data.shape)
	data= F.interpolate(data, size=(112, 112), mode='bilinear', align_corners=False)
	#print(data.shape)
	model_r = ArcFace()
	fea = model_r.forward(data*255)
	return fea

