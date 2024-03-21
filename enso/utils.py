import numpy as np


def sort_arrays(key, *args):

    ind = np.argsort(key)
    key = key[ind]
    
    output = [] 
    for var in args:
        output.append(var[ind])

    return key, *output


if __name__ == "__main__":
    arr1 = np.array([1, 3, 15, 6])
    arr2 = np.arange(3, 11, 2)
    arr3 = np.arange(2, 10, 2)

    print(sort_arrays(arr1, arr2, arr3))
    print()
    print(arr1, arr2, arr3)
