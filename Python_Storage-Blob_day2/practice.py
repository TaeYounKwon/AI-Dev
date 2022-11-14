import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Blob 서비스에 연결합니다. 
connect_string = 'DefaultEndpointsProtocol=https;\
                  AccountName=#;\
                  AccountKey=#;\
                  EndpointSuffix=core.windows.net'

blob_service_client = BlobServiceClient.from_connection_string(connect_string)

# Container 생성한다. 
container_name = str(uuid.uuid4())
print(container_name)

container_client = blob_service_client.create_container(container_name)

# 데이터를 업로드 한다. 

local_path ='./data'
os.mkdir(local_path)

local_file_name = str(uuid.uuid4()) + '.txt'
upload_file_path = os.path.join(local_path, local_file_name)

# 업로드 할 파일을 준비한다. 
file = open(file=upload_file_path, mode='w')
file.write('Hello Azure Storage')
file.close()

blob_client = blob_service_client.get_blob_client(container=container_name,
                                                  blob=local_file_name)

# 파일 업로드 
with open(file=upload_file_path, mode='rb') as data:
  blob_client.upload_blob(data)
  
# 파일의 목록의 확인 
blob_list = container_client.list_blobs()
for blob in blob_list:
  print('\t' + blob.name)
  
#업로드된 파일의 다운로드 

download_file_path = os.path.join(local_path, 
                                  str.replace(local_file_name,'.txt','DOWNLOAD.txt'))
container_client = blob_service_client.get_container_client(container= container_name) 

with open(file=download_file_path, mode='wb') as download_file:
  download_file.write(container_client.download_blob(blob.name).readall())
 
# 실습한 자원의 정리
print('Press the Enter key to begin clean up')
input()

print('Deleting blob container...')
container_client.delete_container()

print('Deleting the local source and downloaded files...')
os.remove(upload_file_path)
os.remove(download_file_path)
os.rmdir(local_path)

print('Done')  