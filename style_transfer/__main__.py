import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image
from .learn import StyleTransfer

def main():
    parser = ArgumentParser(description=("Creates artwork from content and "
                            "style image."),
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('content', metavar='<content path>',
                        help='Content image.')
    parser.add_argument('style', metavar='<style path>', help='Style image.')
    parser.add_argument('--artwork', metavar='<path>',
                        default='artwork.png',
                        help='Save artwork as <path>.')
    parser.add_argument('--init_img', metavar='<path>', 
                        help=('Initialize artwork with image at <path> '
                        'instead of content image.'))
    parser.add_argument('--init_random', action='store_true',
                        help=('Initialize artwork with random image '
                        'instead of content image.'))
    parser.add_argument('--area', metavar='<int>', default=512, type=int, 
                        help=("Content and style are scaled such that their "
                        "area is <int> * <int>. Artwork has the same shape "
                        "as content."))
    parser.add_argument('--iter', metavar='<int>', default=500, type=int, 
                        help='Number of iterations.')
    parser.add_argument('--lr', metavar='<float>', default=1, type=float, 
                        help='Learning rate of the optimizer.')
    parser.add_argument('--content_weight', metavar='<int>', default=1,
                        type=int, help='Weight for content loss.')
    parser.add_argument('--style_weight', metavar='<int>', default=1000,
                        type=int, help='Weight for style loss.')
    parser.add_argument('--content_weights', metavar='<str>',
                        default="{'relu_4_2':1}",
                        help=('Weights of content loss for each layer. '
                        'Put the dictionary inside quotation marks.'))
    parser.add_argument('--style_weights', metavar='<str>',
                        default=("{'relu_1_1':1,'relu_2_1':1,"
                        "'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"),
                        help=('Weights of style loss for each layer. '
                        'Put the dictionary inside quotation marks.'))
    parser.add_argument('--avg_pool', action='store_true',
                        help='Replace max-pooling by average-pooling.')
    parser.add_argument('--no_feature_norm', action='store_false',
                        help=("Don't divide each style_weight by the square "
                        "of the number of feature maps in the corresponding "
                        "layer."))
    parser.add_argument('--preserve_color', choices=['content','style','none'],
                        default='style', help=("If 'style', change content "
                        "to match style color. If 'content', vice versa. " 
                        "If 'none', don't change content or style."))
    parser.add_argument('--weights', choices=['original','normalized'],
                        default='original', help=("Weights of VGG19 Network. "
                        "Either 'original' or 'normalized' weights."))
    parser.add_argument('--device', choices=['cpu','cuda','auto'],
                        default='auto', help='Device used for training.')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision for training.')
    parser.add_argument('--use_adam', action='store_true',
                        help='Use Adam instead of LBFGS optimizer.')
    parser.add_argument('--optim_cpu', action='store_true',
                        help=('Optimize artwork on CPU to move some data from'
                        ' GPU memory to working memory.'))
    parser.add_argument('--quality', metavar='<int>', default=95, type=int,
                        help=('JPEG image quality of artwork, on a scale '
                        'from 1 to 95.'))
    parser.add_argument('--logging', metavar='<int>', default=50, type=int,
                        help=('Number of iterations between logs. '
                        'If 0, no logs.'))
    parser.add_argument('--seed', metavar='<int>', default='random',
                        help='Seed for random number generators.')
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
                                   preserve_color=
                                   args.preserve_color.replace('none',''),
                                   device=args.device,
                                   use_amp=args.use_amp,
                                   adam=args.use_adam,
                                   optim_cpu=args.optim_cpu, 
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
 
if __name__ == '__main__':
    main()
