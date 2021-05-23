## Installation
```bash
pip install git+https://github.com/moritztng/prism.git
```

## Quickstart
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zPlJUYNkmEllnUaZw20GFoYoDawIEtC5?usp=sharing)
[![Read the Docs](https://img.shields.io/badge/Read-Docs-blue?logo=read-the-docs&logoColor=white)](https://prism-style-transfer.readthedocs.io/en/latest/)
#### Minimal Example
```bash
style-transfer content.jpg style.jpg
```
#### Complex Example
```bash
style-transfer content.jpg style.jpg --artwork artwork.png --style_weight 1000 --lr 1 --iter 500
```
<div>
  <img src='images/content.jpg' alt='Content' height='330px'>
  <img src='images/style.jpg' alt='Style' height='330px'><br>
  <img src='images/artwork.jpg' alt='Artwork' width='500px'>
</div>

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

## Shop
Buy an artwork on [prism.art](https://prism.art).

## References
* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)
* [Image Style Transfer using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
* [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/pdf/1606.05897.pdf)
* [Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/pdf/1611.07865.pdf)
* [Supplementary Material: Controlling Perceptual Factors in Neural Style Transfer](http://bethgelab.org/media/uploads/stylecontrol/supplement/)
* [Mixed Precision Training](https://arxiv.org/pdf/1710.03740.pdf)
* [PyTorch Implementation Style Transfer](https://github.com/leongatys/PytorchNeuralStyleTransfer)
* [PyTorch Tutorial Style Transfer](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
* [Palette Knife Painting Sophia by Briahna Wenke](https://www.artbybri.com/2018?pgid=kma1d0tc-db6f04f5-e3a6-4d43-9901-06d9212570ff)
