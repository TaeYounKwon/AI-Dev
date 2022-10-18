import requests
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import json

#Add Subscription Key
subscription_key = '#'
#API url + /face/v1.0/detect
face_api_url='#'

#Class, library, Package 대문자 관례
#지역변수, parameter 소문자 관례
#addr ,msg 줄임말은 배제
#두 단어가 합쳐지면 두번째 단어는 대문자
#상수는 전체가 대문자 const MAX_USER = 100


#Draw box and check how much they are smiling
def DrawBox(faces):

  for face in faces:
    rect = face['faceRectangle']
    left = rect['left']
    top = rect['top']
    width = rect['width']
    height = rect['height']

    draw.rectangle(((left,top),(left+width,top+height)),outline='red')

    face_attributes = face['faceAttributes']
    smile = face_attributes['smile']
    draw.text((left,top),str(smile),fill='red')



#무한도전 맴버들
image_url = 'https://mblogthumb-phinf.pstatic.net/20160913_5/genuinely8_1473773745715iqWzQ_JPEG/20160913_220117.jpg?type=w800'

image = Image.open(BytesIO(requests.get(image_url).content))
image
# image <- view image

headers = {'Ocp-Apim-Subscription-Key': subscription_key}

params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'Smile'
}

data = {'url': image_url}

response = requests.post(face_api_url, params=params, headers=headers,json=data)
faces = response.json()

draw = ImageDraw.Draw(image)
DrawBox(faces)


image