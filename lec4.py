# soft copy
a = [1, 2, 3]

b = a

a[1] = 4
a
b

id(a)
id(b)

# deep copy
a = [1, 2, 3]
a

b = a[:]

a[1] = 4
a
b
