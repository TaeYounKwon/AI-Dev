import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO

#Add subsciprion key
subscription_key = ''
#Add base_url + '/vision/v2.0/
vision_base_url = ''
ocr_url = vision_base_url + 'ocr'


image_url = 'https://parade.com/.image/t_share/MTkwNTc1ODc5NDk1MjMwNTg5/life-quotes-happy.jpg'
image_url_kr = 'https://mblogthumb-phinf.pstatic.net/MjAyMDA2MzBfMjEg/MDAxNTkzNDg5MDEyMjcw.Z2wW_2sz_dI0c09Eggmcu1mlW_YqZXvUF0oUJT0_M_gg.f9jBLRa57oemJjVwc1IX5DI8AK_JiAs3yp2_6MkpBp4g.JPEG.kimss3k/1_%EA%B8%8D%EC%A0%95%EB%AA%85%EC%96%B8_(1).jpg?type=w800'

#Check the Image
image = Image.open(BytesIO(requests.get(image_url).content))


headers = {'Ocp-Apim-Subscription-Key':subscription_key}
params = {'language': 'unk', 'detectOrientation':'true'}
data = {'url':image_url_kr}

response = requests.post(ocr_url,headers=headers, params=params, json=data)
result = response.json()
#result
# {'language': 'en',
#  'textAngle': 0.0,
#  'orientation': 'Up',
#  'regions': [{'boundingBox': '381,109,442,654',
#    'lines': [{'boundingBox': '515,109,178,64',
#      'words': [{'boundingBox': '515,109,178,64', 'text': '"The'}]},
#     {'boundingBox': '438,211,329,63',
#      'words': [{'boundingBox': '438,211,329,63', 'text': 'purpose'}]},
#     {'boundingBox': '381,280,442,63',
#      'words': [{'boundingBox': '381,280,78,63', 'text': 'of'},
#       {'boundingBox': '485,296,137,47', 'text': 'our'},
#       {'boundingBox': '643,280,180,63', 'text': 'lives'}]},
#     {'boundingBox': '461,365,283,63',
#      'words': [{'boundingBox': '461,365,61,63', 'text': 'is'},
#       {'boundingBox': '547,370,77,58', 'text': 'to'},
#       {'boundingBox': '648,365,96,63', 'text': 'be'}]},
#     {'boundingBox': '457,448,292,82',
#      'words': [{'boundingBox': '457,448,292,82', 'text': 'happy."'}]},
#     {'boundingBox': '490,576,211,19',
#      'words': [{'boundingBox': '490,585,19,3', 'text': '—'},
#       {'boundingBox': '518,576,77,19', 'text': 'DALAI'},
#       {'boundingBox': '602,576,73,19', 'text': 'LAMA'},
#       {'boundingBox': '682,585,19,3', 'text': '—'}]},
#     {'boundingBox': '523,732,143,31',
#      'words': [{'boundingBox': '523,732,143,31', 'text': 'Parade'}]}]}]}


for region in result['regions']:
    lines = region['lines']
    
    for line in lines:
        words = line['words']
        
        for word in words:
            print (word['text']) 