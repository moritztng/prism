from flask import Flask, send_file, request
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
    form, files = request.form, request.files
    iter_low, iter_high = int(form['iter_lowres']), int(form['iter_highres'])
    with Image.open(f"images/styles/{form['style']}.jpg") as style, \
         Image.open(files['content']) as content:
        time_low, time_high = time(), 0
        artwork = style_transfer(content, style,
                                area=int(form['area_lowres']),
                                init_random=form['init_random']=='True',
                                iter=iter_low)
        time_low = time() - time_low
        if iter_high:
            with artwork:
                time_high = time()
                artwork = style_transfer(content, style,
                                         area=int(form['area_highres']),
                                         init_img=artwork,
                                         iter=iter_high)
                time_high = time() - time_high
    artwork_bytes = BytesIO()
    with artwork:
        artwork.save(artwork_bytes, format='JPEG',
                     quality=int(form['quality']))
    artwork_bytes.seek(0)
    print(f'{datetime.now().strftime("%d-%m-%Y %H:%M:%S"):*^50}\nLow Res Time '
          f'Total: {time_low:.1f}, Iter: {time_low/iter_low:.3f} | High Res '
          f'Time Total: {time_high:.1f}, Iter: {time_high/max(1,iter_high):.3f}'
          f'\nArguments: {form.to_dict()}')
    return send_file(artwork_bytes, 'image/jpeg')
