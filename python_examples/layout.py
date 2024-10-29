import pycute

d8 = pycute.Layout((8,))
print(d8)

s2xd4 = pycute.Layout((2, 4))
print(s2xd4)

s2xd4_a = pycute.Layout((2, 4), (12, 1))
print(s2xd4_a)

s2xh4 = pycute.Layout((2, (2, 2)), (4, (2, 1)))
print(s2xh4)

# crd2idx
for i in range(s2xh4.size()):
    print(f"i: {i}, s2xh4(i): {s2xh4(i)}")


shape = (3, (2, 3))
print(pycute.prefix_product(shape))
# print(pycute.idx2crd(16, shape))
print(pycute.idx2crd((1, 5), shape))
# print(pycute.idx2crd((1, (1, 2)), shape))

stride = (3, (12, 1))
print(pycute.crd2idx(16, shape, stride))
print(pycute.crd2idx((1, 5), shape, stride))
print(pycute.crd2idx((1, (1, 2)), shape, stride))


print(s2xh4)
print(s2xh4((None, (1, None))))
print(s2xh4((None, (1, 1))))
print(s2xh4((1, (None, 1))))
print(s2xh4((1, None)))

a = pycute.Layout(3, 1)
b = pycute.Layout(4, 3)
row = pycute.make_layout(a, b)
print(row)

a = pycute.Layout((2, 3, 5, 7))
print(a)





# from functools import reduce
# from itertools import chain
# from typing import Union


# def is_tuple(x):
#   return isinstance(x, tuple)

# def product(a):
#   if is_tuple(a):
#     return reduce(lambda val,elem : val*product(elem), a, 1)
#   else:
#     return a

# def idx2crd(idx, shape):

#   if is_tuple(idx):
#     if is_tuple(shape):                # tuple tuple tuple
#       assert len(idx) == len(shape)
#       return tuple(idx2crd(i, s) for i, s in zip(idx,shape))
#     else:                              # tuple "int" "int"
#       assert False           # Error
#   else:
#     if is_tuple(shape):                # "int" tuple tuple
#       tuple_idx = []
#       for i in range(len(shape)):
#         tuple_idx.append(idx2crd(idx // product(shape[:i]), shape[i]))
#       return tuple(tuple_idx)
#     else:                              # "int" "int" "int"
#       return idx % shape

# print("-----------")
# print(idx2crd(16, shape))
# print(idx2crd((1, 5), shape))
# print(idx2crd((1, (1, 2)), shape))


# shape = ((3,), (2,), (3,))

# for idx in range(18):
#     print(f"idx: {idx}, idx2crd(idx, shape): {idx2crd(idx, shape)}")

# # for idx in range(18):
# #     coord_1 = idx % 3
# #     coord_2 = idx // 3
# #     print(f"idx: {(coord_1, coord_2)}, idx2crd(idx, shape): {idx2crd((coord_1, coord_2), shape)}")

# for idx in range(18):
#     coord_1 = idx % 3
#     coord_2 = (idx // 3) % 2
#     coord_3 = idx // 6
#     print(f"idx: {(coord_1, coord_2, coord_3)}, idx2crd(idx, shape): {idx2crd((coord_1, coord_2, coord_3), shape)}")