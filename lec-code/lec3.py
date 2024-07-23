# 데이터 타입
x = 15
print(x, "는", type(x), "형식입니다.", sep=':)')


# 문자형 데이터 예제
a = "Hello, World!"
b = "python programming"

print(a, type(a))
print(b, type(b))


# 여러줄 문자열
ml_str = """This is
a multi-line
string"""

print(ml_str, type(ml_str))


# 문자열 연산
greeting = "안녕" + " " + "파이썬!"


# 리스트
fruit = ['apple', 'banana', 'cherry']
type(fruit)

numbers = [1,2,3,4,5]
type(numbers)

mixed_list = [1, 'hello', [1, 2, 3]]
type(mixed_list)


# 튜플
a = (10, 20, 30) # a = 10, 20, 30 과 동일 (괄호 생략하고 쉼표(,)로 구분해도 튜플 생성이 가능)
b = (42,)
c = (10)
d = 1, 2, 3

a
type(a)

b
type(b)

c
type(c)

d
type(d)

## 튜플의 인덱싱과 슬라이싱
a_list = [10, 20, 30, 40, 50]
a_tuple = (10, 20, 30, 40, 50)

a_list[1] = 25
a_tuple[1] = 25 # 튜플 값은 변경이 불가하다

a_list
a_list[1:]
a_list[:3]
a_list[0:3]

a_tuple
a_tuple[1:] # 두 번째 원소부터 마지막 원소까지 슬라이싱, 해당 인덱스 이상
a_tuple[:3] # 첫 번째 원소부터 네 번째 원소까지 슬라이싱, 해당 인덱스 미만
a_tuple[1:3] # 1, 2번 인덱스 슬라이싱, 해당 인덱스 이상&미만


## 사용자 정의 함수
def min_max(numbers):
 return min(numbers), max(numbers)

a = [1, 2, 3, 4, 5]
result = min_max(a)
result
type(result)

print("Minimum and maximum:", result)


# 딕셔너리
person = {
 'name': 'John',
 'age': 30,
 'city': 'New York'
}

print("Person:", person)

summer = {
  "name": "Summer",
  "age": (28, 24),
  "city": ["Seoul", "Daegu"]
}

print("summer:", summer)
print("summer:", summer["name"])

summer.get('name')
summer.get('age')

summer.get('age')[0]

summer_city = summer.get('city')
summer_city[0]


# 집합형(set)
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits) # 중복 'apple'은 제거됨

type(fruits)

## 빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add("apple")
empty_set.add("banana")
empty_set.add("apple")

empty_set.remove("banana")
empty_set.discard("cherry") # 집합에 존재하지 않는 요소를 삭제해도 에러 발생 X
empty_set.remove("cherry") # 집합에 존재하지 않는 요소를 삭제하면 KeyError 발생

empty_set

## 집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) # union: 합집합
intersection_fruits = fruits.intersection(other_fruits) # intersection: 교집합
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)


# 논리형 데이터 타입
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p) # True는 1로, False는 0으로 계산됩니다.

is_active = True
is_greater = 10 > 5 # True 반환
is_equal = (10 == 5) # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

a = 3
if (a == 2): # if문 괄호 안에 bool값을 가지는 논리형 값이 들어가야 함
 print("a는 2와 같습니다.")
else:
 print("a는 2와 같지 않습니다.")
 
 
# 데이터 타입 변환
## 숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

# 문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

# 리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst, type(lst))
tup = tuple(lst)
print("튜플:", tup, type(tup))

# 집합(set)과 딕셔너리
set_example = {'a', 'b', 'c'}
# 딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example} # key값은 True로 사용할 거고, key값은 set_example에서 가져와
print("Dictionary from set:", dict_from_set, type(dict_from_set))
