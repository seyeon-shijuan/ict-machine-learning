# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:43:35 2023

@author: 예비
"""


# 산술연산
3.14 * 10 * 10
3.14 * 10**2
#자료형
type(10)
type(3.14)
type("python")

r = 20
PI = 3.14 # 원주율 정의
area = PI * r**2
area

lst = [ 10, 20, 30, 40, 50] # 리스트 정의
lst # 리스트 출력
lst[2] # 리스트의 원소 접근
lst[2] = 90 # 세 번째 원소를 90으로 변경
lst # 리스트 출력
len(lst)

lst[0:3] # 인덱스 0부터 2까지를 추출한다.
lst[2:] # 인덱스 2부터 끝까지를 추출한다.
lst[:3] # 인덱스 0부터 2까지를 추출한다.
lst[:-1] # 처음부터 마지막 원소 앞까지 추출한다.

car = { 'HP':200, 'make': "BNW" } # 딕셔너리 정의
car['HP'] # 원소에 접근
car['color'] = "white" # 새 원소 추가
car

temp = -10
if temp < 0 :
    print("영하입니다.")
else :
    print("영상입니다.")

for i in [1, 2, 3, 4, 5] :
    print(i, end=" ")

#리스트 내포 기법 사용방법
#for문에 if문
a = [3, 7, 9, 10] 
result = [i + 10 for i in a if i%2 != 0]

#이중 for문
result = [] 
for i in range(1, 10):     
	for j in range(1, 10):
		result.append(i * j)  
        
result1 = [i * j for i in range(1, 10) for j in range(1, 10)]
#이중 for문에 if문
result = [] 
for x in range(1, 10): 
	if x%2 == 0: 
		for y in range(1, 10): 
			result.append(x * y)  
result2 = [x * y for x in range(1, 10) if x%2 == 0 for y in range(1, 10)]


'''class'''

class Pen:
    def __init__(self, name, color, shape, price):
        self.name = name
        self.color = color
        self.shape = shape
        self.price = price
    
    def explain(self):
        print(f"이름: {self.name}, 색상: {self.color}, 모양: {self.shape}, 가격:{self.price}")


pen = Pen("모나미","파랑","빨대",3000)        
pen.explain()



        
        
        