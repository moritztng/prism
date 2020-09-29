from argparse import ArgumentParser
import requests

parser = ArgumentParser(description="Test Flask App")
parser.add_argument('--url', default='http://127.0.0.1:5000/style_transfer')
parser.add_argument('--content', default='images/contents/content.jpg')
parser.add_argument('--style', default='style')
parser.add_argument('--init_random', default='False')
parser.add_argument('--iter_lowres', default='500')
parser.add_argument('--area_lowres', default='512')
parser.add_argument('--iter_highres', default='200')
parser.add_argument('--area_highres', default='1024')
parser.add_argument('--quality', default='95')
args = parser.parse_args()

data = vars(args)
url = data.pop('url')
content = data.pop('content')
with open(content, 'rb') as content, open('artwork.jpg', 'wb') as artwork:
    r = requests.post(url, data=data, files={'content': content})
    artwork.write(r.content)
