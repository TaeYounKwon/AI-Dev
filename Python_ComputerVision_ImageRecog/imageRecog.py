# 1. Image Tagging
# Computer Vision Object Dectection
# Using Computer Vision API to classify the objects in the image

# Using 'requests' package to connet to network system.
import requests

# 이미지처리를 위해서 matplotlib.pyplot, Image, BytesIO 세 개의 패키지를 import 합니다.

# matplotlib.pyplot는 import 할 때 시간이 조금 걸릴 수 있습니다.
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import json

# Function to implement the Bounding Box
# Draw blue lines to every recognized objects
# Write their object name
def drawBox(detectData):
    objects = detectData['objects']
    
    for obj in objects:
        #print(obj)

        rect = obj['rectangle']
        #print(rect)
        
        x = rect['x']
        y = rect['y']
        w = rect['w']
        h = rect['h']
        
        draw.rectangle(((x,y),(x+w,y+h)),outline='blue')
        
        objectName = obj['object']
        draw.text((x,y),objectName,fill='red')
        
        
        


# Set up the Subscription Key& necessary URL (vision/v2.0 -> 버젼 2.0 기반 API 사용).
subscription_key = '4963557778fe4e7f89930bba0ae1cdeb'
vision_base_url = 'https://labuser50computervision.cognitiveservices.azure.com/vision/v2.0/'

#컴퓨터 비젼은 객체 감지(Object detection) 와 객체 분류&분석을 사용할 수 있음
#객체 감지는 rectangle로 나옴 
analyze_url = vision_base_url + 'analyze'

#분석에 사용되는 이미지를 확인 합니다.
#Sample image = Time Square
image_url = 'https://img.kr.news.samsung.com/kr/wp-content/uploads/2021/07/210728waterfall1.jpg'
con = requests.get(image_url).content
byte = BytesIO(con)
image = Image.open(byte)
# or simply write -> image=Image.open(BytesIo(requests.get(image_url).content))

headers = {'Ocp-Apim-Subscription-key': subscription_key}
params = {'visualFeatures' : 'Categories,Description,Color'}
data = {'url': image_url}

response = requests.post(analyze_url, headers = headers, params = params, json = data) # get or post

result = response.json()


#Output
# {'categories': [{
#     'name': 'outdoor_street', 
#     'score': 0.67578125, 
#     'detail': {
#         'landmarks': [{
#             'name': 'Times Square', 'confidence': 0.962982714176178}]}}], 
#  'color': {
#      'dominantColorForeground': 'White', 
#      'dominantColorBackground': 'Black', 
#      'dominantColors': ['Black'], 
#      'accentColor': '206CAB', 
#      'isBwImg': False, 
#      'isBWImg': False}, 
#  'description': {
#      'tags': ['building', 'outdoor', 'street', 'city', 'night', 'filled', 'many', 'table', 'sign', 'walking', 'people', 'man', 'holding', 'store', 'snow', 'tall', 'standing', 'traffic', 'group'], 
#      'captions': [{'text': 'Times Square street', 'confidence': 0.8915447555816607}]}, 
#  'requestId': '46ac60f0-a4d6-4646-ad70-cb372300a5bc', 
#  'metadata': {'height': 563, 'width': 1000, 'format': 'Jpeg'}}


#How to print out the detailed data from Json file.
image_caption = result['description']['captions'][0]['text']
#print(image_caption)


# 2. Object Detection

object_Detection_url = vision_base_url + 'detect'
image_url2 = 'https://health.chosun.com/site/data/img_dir/2018/01/17/2018011700908_0.jpg'

image2 = Image.open(BytesIO(requests.get(image_url2).content))

headers = {'Ocp-Apim-Subscription-key': subscription_key}
params = {'visualFeatures' : 'Categories,Description,Color'}
data = {'url': image_url2}

response = requests.post(object_Detection_url, headers = headers, params = params, json = data)
result2 = response.json()
#print(result2)

# {'objects': [
#     {'rectangle':     {'x': 334, 'y': 95, 'w': 84, 'h': 142}, 
#      'object': 'dog', 
#      'confidence': 0.65, 
#      'parent': {
#          'object': 'mammal', 
#          'confidence': 0.708, 
#          'parent': {
#              'object': 
#                  'animal', 
#                  'confidence': 0.708}}},
#     {'rectangle': {'x': 415, 'y': 73, 'w': 101, 'h': 199}, 
#      'object': 'retriever', 
#      'confidence': 0.632, 
#      'parent': {'object': 'dog', 
#                 'confidence': 0.705, 
#                 'parent': {'object': 'mammal', 
#                            'confidence': 0.759, 
#                            'parent': {'object': 'animal', 
#                                       'confidence': 0.76}}}}, 
#     {'rectangle': {'x': 505, 'y': 52, 'w': 92, 'h': 236}, 
#      'object': 'mammal', 
#      'confidence': 0.877, 
#      'parent': {'object': 'animal', 
#                 'confidence': 0.878}},
#     {'rectangle': {'x': 0, 'y': 141, 'w': 56, 'h': 132}, 
#      'object': 'Angora rabbit', 
#      'confidence': 0.657, 
#      'parent': {'object': 'rabbit', 
#                 'confidence': 0.714, 
#                 'parent': {'object': 'mammal', 
#                            'confidence': 0.801,
#                            'parent': {'object': 'animal', 
#                                       'confidence': 0.806}}}},
#     {'rectangle': {'x': 63, 'y': 93, 'w': 117, 'h': 182}, 
#      'object': 'retriever', 
#      'confidence': 0.564, 
#      'parent': {'object': 'dog',
#                 'confidence': 0.614, 
#                 'parent': {'object': 'mammal', 
#                            'confidence': 0.797, 
#                            'parent': {'object': 'animal', 
#                                       'confidence': 0.799}}}}, 
#     {'rectangle': {'x': 145, 'y': 131, 'w': 98, 'h': 153},
#      'object': 'dog', 
#      'confidence': 0.727, 
#      'parent': {'object': 'mammal', 
#                 'confidence': 0.801, 
#                 'parent': {'object': 'animal', 'confidence': 0.802}}},
#     {'rectangle': {'x': 279, 'y': 146, 'w': 68, 'h': 122}, 
#      'object': 'mammal', 
#      'confidence': 0.529, 
#      'parent': {'object': 'animal', 'confidence': 0.529}},
#     {'rectangle': {'x': 396, 'y': 194, 'w': 102, 'h': 93}, 
#      'object': 'mammal', 
#      'confidence': 0.812, 
#      'parent': {'object': 'animal', 'confidence': 0.812}},
#     {'rectangle': {'x': 242, 'y': 211, 'w': 66, 'h': 72}, 
#      'object': 'mammal',
#      'confidence': 0.668, 
#      'parent': {'object': 'animal', 'confidence': 0.67}},
#     {'rectangle': {'x': 314, 'y': 213, 'w': 85, 'h': 70}, 
#      'object': 'mammal', 
#      'confidence': 0.607, 
#      'parent': {'object': 'animal', 
#                 'confidence': 0.607}}], 
#  'requestId': '54c5795d-d398-419a-a16d-aad433d9304c', 
#  'metadata': {'height': 299, 'width': 600, 'format': 'Jpeg'}}

# Draw Sqaure lines from object detected image

draw = ImageDraw.Draw(image2)

drawBox(result2)


image2