import random

random_num = random.randint(1,100)  #1에서 100사이

count = 1
#print(random_num)
while True:

        my_num = int(input("1에서 100사이의 숫자를 입력하세요. : "))
        
        if my_num>random_num:
            print("down")
        elif my_num<random_num:
            print("up")
        else:
            print(f"축하합니다. {count}번만에 맞췄습니다!")
            break
        count +=1
