

from dask.distributed import Client, progress

client = Client('tcp://hpg6-112.maas:8786', threads_per_worker=4, n_workers=10)
# client = Client('hpg6-112.maas:8786', threads_per_worker=4, n_workers=3)
# client = Client('tcp://192.168.10.113:34705',
#                 threads_per_worker=4, n_workers=3)
print client

from time import sleep


def inc(x):
    sleep(1)
    return x + 1


def add(x, y):
    sleep(1)
    return x + y


# This takes three seconds to run because we call each
# function sequentially, one after the other

x = inc(1)
y = inc(2)
z = add(x, y)


from dask import delayed

# x = delayed(inc)(1)
# y = delayed(inc)(2)
# z = delayed(add)(x, y)
# print z.compute()


# Parallelize

# data = [1, 2, 3, 4, 5, 6, 7, 8]
data = range(50)

results = []
for x in data:
    y = delayed(inc)(x)
    results.append(y)

total = sum(results)
print total.compute()
