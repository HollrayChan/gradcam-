import argparse
"""
    base Configuration
"""
parser = argparse.ArgumentParser(description='MGN')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs')
parser.add_argument("--datadir", type=str, default="Market-1501-v15.09.15", help='dataset directory')

parser.add_argument('--load_sel', type=str, default='', help='load weights selectively from a trained model')
parser.add_argument('--model', default='MGN', help='weight name')
parser.add_argument('--save', type=str, default='test', help='file name to save')

parser.add_argument('--cam_num', type=int, default=300, help='cam_num pairs')
parser.add_argument("--layer_name", type=str, default='p1', help='layer_name in your module')
parser.add_argument('--height', type=int, default=384, help='height of the input image')
parser.add_argument('--width', type=int, default=128, help='width of the input image')
parser.add_argument('--similarly', action='store_true',  default=False, help='output similarly in img')
"""
    model Configuration
"""
# mgn
parser.add_argument('--module', type=str, default='MGN', help='choose between RPP training mode and MGN normal training mode')
parser.add_argument('--slice_p1', type=int, default=2, help='slice p2 24*8 to how many horizontal slices')
parser.add_argument('--slice_p2', type=int, default=2, help='slice p2 24*8 to how many horizontal slices')
parser.add_argument('--slice_p3', type=int, default=3, help='slice p3 24*8 to how many horizontal slices')
parser.add_argument('--feats', type=int, default=256, help='number of feature maps')
parser.add_argument('--num_classes', type=int, default=751, help='')
"""
    Loss Configuration
"""
parser.add_argument('--gradcam_loss', type=str, default='1*CrossEntropy', help='gradcam_loss')
# triplet
parser.add_argument("--margin", type=float, default=1.2, help='')
# centerloss
parser.add_argument("--center", type=str, default='yes', help='center loss')
parser.add_argument("--center_lr", type=float, default=0.5, help='Learning rate of SGD to learn the centers of center loss')
parser.add_argument("--center_loss_weight", type=float, default=0.0005, help='')
# circleloss
parser.add_argument('--loss_pretrain', action='store_true',  default=False, help='only use crossentropy in the first 500 iter')
parser.add_argument('--circle_m', type=float, help='m for circle loss', default=0.3)
parser.add_argument('--circle_gamma',type=float, default=60, help='gamma for circle loss')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

