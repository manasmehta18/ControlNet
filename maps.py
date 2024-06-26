from annotator.util import resize_image, HWC3
import cv2 as cv
import os

model_canny = None

def canny(img, res, l, h):
    img = resize_image(HWC3(img), res)
    global model_canny
    if model_canny is None:
        from annotator.canny import CannyDetector
        model_canny = CannyDetector()
    result = model_canny(img, l, h)
    return result

model_hed = None


def hed(img, res):
    img = resize_image(HWC3(img), res)
    global model_hed
    if model_hed is None:
        from annotator.hed import HEDdetector
        model_hed = HEDdetector()
    result = model_hed(img)
    return result


model_mlsd = None


def mlsd(img, res, thr_v, thr_d):
    img = resize_image(HWC3(img), res)
    global model_mlsd
    if model_mlsd is None:
        from annotator.mlsd import MLSDdetector
        model_mlsd = MLSDdetector()
    result = model_mlsd(img, thr_v, thr_d)
    return result


model_openpose = None


def openpose(img, res, has_hand):
    img = resize_image(HWC3(img), res)
    global model_openpose
    if model_openpose is None:
        from annotator.openpose import OpenposeDetector
        model_openpose = OpenposeDetector()
    result, _ = model_openpose(img, has_hand)
    return result


model_uniformer = None


def uniformer(img, res):
    img = resize_image(HWC3(img), res)
    global model_uniformer
    if model_uniformer is None:
        from annotator.uniformer import UniformerDetector
        model_uniformer = UniformerDetector()
    result = model_uniformer(img)
    return result

# constable, lionel, lee, va, watts, boudin, cox

# bonheur, boudin, constable, courbet, jones, manet

painter = "manet"
source = os.listdir("paintings/" + painter)
dest = os.listdir("maps/" + painter)

count = 1

for img in source:
    src = cv.imread("paintings/" + painter + "/" + img)
    # map = canny(src, 512, 100, 200)
    map = hed(src, 512)
    cv.imwrite(os.path.join("maps/" + painter, img.rsplit( ".", 1 )[ 0 ] + '.jpg'), map)
    count += 1


