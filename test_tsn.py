import argparse
import time
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
import torchvision
from dataset import *
from transforms import *
from model.models import *
from model.ops import ConsensusModule

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'], default="RGB")
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')
parser.add_argument('--context', default=False, action="store_true")
parser.add_argument('--categorical', default=True, action="store_true")
parser.add_argument('--continuous', default=True, action="store_true")

args = parser.parse_args()

model = 'rgb_with_context_tsn.pth.tar' # 0.2157
model = 'flow_tsn.pth.tar' # 0.2213


if args.modality == 'RGB':
    data_length = 1
elif args.modality == 'Flow':
    data_length = 5

args.weights = model
args.test_crops = 1
args.test_segments = 25

print(args)

net = TSN(26, 1, args.modality,
          base_model=args.arch, new_length=data_length,
          consensus_type=args.crop_fusion_type, embed=True, context=args.context,
          dropout=args.dropout)

features_blobs = []

checkpoint = torch.load(args.weights)

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
a = net.load_state_dict(base_dict, strict=True)
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale((224,224)),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(224, 224)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))



data_loader = torch.utils.data.DataLoader(
        TSNDataSet("test", num_segments=args.test_segments, context=args.context,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

devices = [0]

net = torch.nn.DataParallel(net.cuda(), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []

def eval_video(video_data):
    i, data, label, label_cont = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                           volatile=True)

    out = net(input_var, None)
    rst = torch.sigmoid(out['categorical']).data.cpu().numpy().copy()
    rst_cont = torch.sigmoid(out['continuous']).data.cpu().numpy().copy()

    return i, rst.reshape((num_crop, args.test_segments, 26)).mean(axis=0).reshape(
        (args.test_segments, 1, 26)
    ), rst_cont.reshape((num_crop, args.test_segments, 3)).mean(axis=0).reshape(
        (args.test_segments, 1, 3)
    )

proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

import random

for i, batch in data_gen:
    # print(batch)
    data, embeddings = batch

    rst = eval_video((i, data, None, None))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
  
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))

video_pred = np.squeeze(np.array([np.mean(x[0], axis=0) for x in output]))
video_pred_cont = np.squeeze(np.array([np.mean(x[1], axis=0) for x in output]))
print(video_pred.shape, video_pred_cont.shape)
