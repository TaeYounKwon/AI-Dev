import socket

in_addr = socket.gethostbyname(socket.gethostname())
#서버이름을 통해 IP주소 할당    서버이름(컴퓨터이름)

print(in_addr)