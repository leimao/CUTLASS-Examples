import pycute

layout = pycute.Layout((2, (1, 6)), (1, (6, 2)))
print(layout)
print(pycute.coalesce(layout))
# By-mode coalesce
print(pycute.coalesce(layout, profile=(1,1)))

