## Installation
```bash
pip install git+https://github.com/moritztng/prism.git
```

## Quickstart
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zPlJUYNkmEllnUaZw20GFoYoDawIEtC5?usp=sharing)
#### Minimal Example
```bash
style-transfer content.jpg style.jpg
```
#### Complex Example
```bash
style-transfer content.jpg style.jpg --artwork artwork.png --style_weight 1000 --lr 1 --iter 500
```

## Features
#### High Resolution
```bash
# Create low-resolution artwork with area of 512 * 512. 
style-transfer content.jpg style.jpg
# Initialize with low-resolution artwork to create artwork with area of 1024 * 1024. 
style-transfer content.jpg style.jpg --init_img artwork.png --area 1024 --iter 200
```
#### Mixed Precision
Faster training, less memory, same quality on GPUs. 
```bash
style-transfer content.jpg style.jpg --use_amp
```
#### Preserve Content Color
```bash 
style-transfer content.jpg style.jpg --preserve_color content
```

## Python Object
```python
from PIL import Image
from style_transfer.learn import StyleTransfer

style_transfer = StyleTransfer()
artwork = style_transfer(Image.open('content.jpg'), Image.open('style.jpg'))
artwork.save('artwork.png')
```
