from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

#Add training directory ex) https://###.cognitiveservices.azure.com/
ENDPOINT_Training = '#'
#Add prediction directory ex) https://###-prediction.cognitiveservices.azure.com/
ENDPOINT_Prediction = '#'

#Add training and prediction key
training_key = '#' 
prediction_key = '#' 
#Add resource id
prediction_resource_id = '#'

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT_Training, credentials)

print ("Creating project...")
project = trainer.create_project("### Project")

Jajangmyeon_tag = trainer.create_tag(project.id, "Jajangmyeon")
Champon_tag = trainer.create_tag(project.id, "Champon")
Tangsuyug_tag = trainer.create_tag(project.id, "Tangsuyug")

print('Training....')
iteration = trainer.train_project(project.id)
while (iteration.status != 'Completed'):
  iteration = trainer.get_iteration(project.id, iteration.id)
  print('Training status: ' + iteration.status)

  time.sleep(10)
print('-------------------')
print('Training Finished!!')

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT_Prediction, prediction_credentials)

target_image_url = 'https://w.namu.la/s/d4c53737b61fec8cf0fa02206d85a5022fc5465593f2e0190648f7c5911acd836a5f7a1db0f19f0136ec1c178d782465a9455b31d178b79df5133fc6b493a41f8e6fe6815d3da4aafdd31ec6ce870697e8db18daef4fee7a1420b29aab35b3a2'
result = predictor.classify_image_url(project.id, '####project name', target_image_url)

for prediction in result.predictions:
  print('\t' + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))