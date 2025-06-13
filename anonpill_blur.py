from PIL import Image,ImageOps,ImageDraw,ImageFilter,ImageFont
import cv2
import numpy as np
from ultralytics import YOLO
import math

resW = 640
resH = 640
 
def detect(img,facmodel,licpmodel,faces=True,licenseplates=True):
    detections = []
    labels = []
    #frame =  np.array(img) 

    if faces:
        results = facmodel(img, verbose=False)
        detections.extend(results[0].boxes)
        labels.append('Face')

    if licenseplates:
        results = licpmodel(img, verbose=False)
        detections.extend(results[0].boxes)
        labels.append('License Plate')

    return detections,labels,img

def pointsDistance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def blurRectangle(img,p0,p1):
    # Create rectangle mask
    mask = Image.new('L', im.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([ p0,p1], fill=255)

    # Apply the GaussianBlur filter to the entire image
    blurred_image = img.filter(ImageFilter.GaussianBlur(radius=5))

    # Apply the mask to the blurred image
   # blurred_region = ImageOps.invert(mask)
    img.paste(blurred_image, mask=mask)
    #blurred_image = Image.composite(blurred_image, img, blurred_region)
    return img

def showDetections(im,detections,min_thresh,box = True,blur=True,circle=False):

    # Create a drawing object
    draw = ImageDraw.Draw(im)
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > min_thresh:
            #if circle:
                #center = (int((xmin + xmax)/2),int((ymin + ymax)/2))
                #radius = int(pointsDistance(center, (xmax,ymax)))
                #cv2.circle(frame, center, radius, color, 2)
            if blur:
                im = blurRectangle(im,(xmin,ymin),(xmax,ymax))
            if box:
                draw.rectangle(((xmin,ymin), (xmax,ymax)), outline = 'red')
                conf_str = f'{int(conf*100)}%'
                font = ImageFont.truetype('Roboto.ttf', 15)  # You can change the font and size
                draw.text((xmin, ymin), conf_str, fill='red', font=font)

    return im

def getImgDetected(img,facmodel,licpmodel,faces=True,licenseplates=True,box = True,blur=True,min_thresh = 0.5):
    detections,labels,frame = detect(img,facmodel,licpmodel,faces=faces,licenseplates=licenseplates)
    frame = showDetections(frame,detections,min_thresh,box = box,blur=blur)
    return frame

def getImgAnonymized(img,facmodel,licpmodel,faces=True,licenseplates=True,min_thresh = 0.5):
    pass

def resize_image_to_max_width(img, max_width):
    ratio = img.size[1]/img.size[0]
    height = int(ratio*max_width)
    resized_img = img.resize((max_width, height))
    return resized_img

fa_model = YOLO('models/best_faces_yolo11m_100epochs_13-6-2025.pt', task='detect')
li_model = YOLO('models/best_licenseplates_yolo11m_100epochs_11-6-2025.pt', task='detect')

#im = Image.open('test_img1["Cross walk" by docpop is licensed under CC BY-SA 2.0].jpg') 
#im = Image.open('/home/zezo/anonpill/test_img2["Ranger directing traffic at North Entrance" by YellowstoneNPS is marked with Public Domain Mark 1.0].jpg')
im = Image.open('test_img3 ["people-waiting-at-the-bus-station-19143071" by Kelly from Pexel].jpg')
#im = ImageOps.contain(im, (resW,resH))
im = im.convert('RGB')
im = resize_image_to_max_width(im,640)
print("im size: ",im.size)
im_ = getImgDetected(im,fa_model,li_model,box=False,faces=False,licenseplates=False)

im_.show()