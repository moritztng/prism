import torch
from argparse import ArgumentParser
from PIL import Image
from style_transfer.learn import StyleTransfer

parser = ArgumentParser(description="Prism Style Transfer")
parser.add_argument('--content', default='images/contents/content.jpg')
parser.add_argument('--style', default='images/styles/style.jpg')
parser.add_argument('--artwork', default='artwork.png')
parser.add_argument('--init_img')
parser.add_argument('--init_random', action='store_true')
parser.add_argument('--area', default=512, type=int)
parser.add_argument('--iter', default=500, type=int)
parser.add_argument('--lr', default=1, type=int)
parser.add_argument('--content_weight', default=1, type=int)
parser.add_argument('--style_weight', default=1e3, type=int)
parser.add_argument('--content_weights', default="{'relu_4_2':1}")
parser.add_argument('--style_weights', default=("{'relu_1_1':1,'relu_2_1':1,"
                    "'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"))
parser.add_argument('--avg_pool', action='store_true')
parser.add_argument('--no_feature_norm', action='store_false')
parser.add_argument('--preserve_color', default='style')
parser.add_argument('--weights', default='original')
parser.add_argument('--device', default='auto')
parser.add_argument('--quality', default=95, type=int)
parser.add_argument('--logging', default=50, type=int)
parser.add_argument('--seed', default='random')
args = parser.parse_args()

if args.seed != 'random':
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.manual_seed(int(args.seed))

style_transfer = StyleTransfer(lr=args.lr,
                               content_weight=args.content_weight,
                               style_weight=args.style_weight,
                               content_weights=args.content_weights,
                               style_weights=args.style_weights,
                               avg_pool=args.avg_pool,
                               feature_norm=args.no_feature_norm,
                               weights=args.weights,
                               preserve_color=args.preserve_color,
                               device=args.device,
                               logging=args.logging)

init_img = Image.open(args.init_img) if args.init_img else None
with Image.open(args.content) as content, Image.open(args.style) as style:
    artwork = style_transfer(content, style,
                             area=args.area,
                             init_random=args.init_random,
                             init_img=init_img,
                             iter=args.iter)
artwork.save(args.artwork, quality=args.quality)
artwork.close()
if init_img:
    init_img.close()
