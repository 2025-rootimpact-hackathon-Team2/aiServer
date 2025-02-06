from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from . import newPro
import cv2
import matplotlib.pyplot as plt

def test(request):
    fileName = request.POST["fileName"]
    print(fileName)
    img = cv2.imread('C:/uploads/{}'.format(fileName), cv2.IMREAD_COLOR)
    # C:\Users\dmsql\OneDrive\바탕 화면\aiTraining\shop\src\main\resources
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = newPro.pred_img(img)
    return HttpResponse(result)