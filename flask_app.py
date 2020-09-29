from flask import Flask, request, send_file
from PIL import Image
from io import BytesIO
from time import time
from datetime import datetime
from style_transfer.learn import StyleTransfer

app = Flask(__name__)
style_transfer = StyleTransfer(lr=1,
                               content_weight=1,
                               style_weight=1e3,
                               content_weights="{'relu_4_2':1}",
                               style_weights=("{'relu_1_1':1,'relu_2_1':1,"
                               "'relu_3_1':1,'relu_4_1':1,'relu_5_1':1}"),
                               avg_pool=False,
                               feature_norm=True,
                               weights='original',
                               preserve_color='style',
                               device='auto',
                               logging=0)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/style_transfer', methods=['POST'])
def transfer():
    with Image.open(f"images/styles/{request.form['style']}.jpg") as style, \
         Image.open(request.files['content']) as content:
        time_lowres, time_highres = time(), 0
        artwork = style_transfer(content, style,
                                area=int(request.form['area_lowres']),
                                init_random=request.form['init_random']=='True',
                                iter=int(request.form['iter_lowres']))
        time_lowres = time() - time_lowres
        if int(request.form['iter_highres']):
            with artwork:
                time_highres = time()
                artwork = style_transfer(content, style,
                                         area=int(request.form['area_highres']),
                                         init_img=artwork,
                                         iter=int(request.form['iter_highres']))
                time_highres = time() - time_highres
    artwork_bytes = BytesIO()
    with artwork:
        artwork.save(artwork_bytes, format='JPEG',
                     quality=int(request.form['quality']))
    artwork_bytes.seek(0)
    print(f'{datetime.now().strftime("%d-%m-%Y %H:%M:%S"):*^50}\n'
          f'Time Low Res: {time_lowres:.1f} | Time High Res: {time_highres:.1f}'
          f'\nArgs: {request.form.to_dict()}')
    return send_file(artwork_bytes, 'image/jpeg')
