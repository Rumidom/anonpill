from PIL import Image
import anonpill as anon

im = Image.open('docs/test_img1.jpg') 
#im = Image.open('docs/test_img2.jpg')
#im = Image.open('docs/test_img3.jpg')
#im = ImageOps.contain(im, (resW,resH))

im_ = anon.getImgAnonymized(im,box=False,faces=True,licenseplates=True)
im_.show()