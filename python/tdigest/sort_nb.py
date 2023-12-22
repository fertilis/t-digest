"""
Cannot call sort() on numpy array in no-python mode in numba.
So, need to implement it manually.
"""
import numpy as np
import numba as nb


@nb.njit(cache=True)
def insertion_sort_nb(array: "np.ndarray[T]", is_less: "(T) -> bool"):
    for i in range(1, len(array)):
        j = i
        while j > 0 and not is_less(array[j - 1], array[j]):
            array[j - 1], array[j] = array[j], array[j - 1]
            j -= 1

# Quicksort
@nb.njit(cache=True)
def _partition(arr, l, h, is_less_than_or_equal):
    i = l - 1
    x = arr[h]
    for j in range(l, h):
        if is_less_than_or_equal(arr[j], x):
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[h] = arr[h], arr[i + 1]
    return i + 1


@nb.njit(cache=True)
def quick_sort_iterative_nb(arr, is_less_than_or_equal):
    l = 0
    h = len(arr) - 1
    # Create an auxiliary stack
    size = h - l + 1
    stack = np.zeros(size, dtype=np.int64)

    # initialize top of stack
    top = -1

    # push initial values of l and h to stack
    top = top + 1
    stack[top] = l
    top = top + 1
    stack[top] = h

    # Keep popping from stack while is not empty
    while top >= 0:
        # Pop h and l
        h = stack[top]
        top = top - 1
        l = stack[top]
        top = top - 1

        # Set pivot element at its correct position in
        # sorted array
        p = _partition(arr, l, h, is_less_than_or_equal)

        # If there are elements on left side of pivot,
        # then push left side to stack
        if p - 1 > l:
            top = top + 1
            stack[top] = l
            top = top + 1
            stack[top] = p - 1

        # If there are elements on right side of pivot,
        # then push right side to stack
        if p + 1 < h:
            top = top + 1
            stack[top] = p + 1
            top = top + 1
            stack[top] = h


@nb.njit(cache=True)
def quick_sort_recursive_nb(array, is_less_than_or_equal):
    _quick_sort_recursive_nb(array, 0, len(array)-1, is_less_than_or_equal)


@nb.njit(cache=True)
def _quick_sort_recursive_nb(array, low, high, is_less_than_or_equal):
    if low < high:
        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = _partition(array, low, high, is_less_than_or_equal)

        # Recursive call on the left of pivot
        _quick_sort_recursive_nb(array, low, pi - 1, is_less_than_or_equal)

        # Recursive call on the right of pivot
        _quick_sort_recursive_nb(array, pi + 1, high, is_less_than_or_equal)



@nb.njit(cache=True)
def merge_sort_nb(arr, is_less_than_or_equal):
    if len(arr) > 1:
        # Finding the mid of the array
        mid = len(arr)//2
        # Dividing the array elements
        L = arr[:mid]
        # Into 2 halves
        R = arr[mid:]
        # Sorting the first half
        merge_sort_nb(L, is_less_than_or_equal)
        # Sorting the second half
        merge_sort_nb(R, is_less_than_or_equal)
        i = j = k = 0
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if is_less_than_or_equal(L[i], R[j]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1




if __name__ == "__main__":
    import time

    @nb.njit(cache=True)
    def is_less(x: np.float64, y: np.float64):
        return x < y

    @nb.njit(cache=True)
    def is_less_than_or_equal(x: np.float64, y: np.float64):
        return x <= y

    array = np.random.random(100)

    # Caching numba compilations
    insertion_sort_nb(array.copy(), is_less)
    quick_sort_iterative_nb(array.copy(), is_less_than_or_equal)
    quick_sort_recursive_nb(array.copy(), is_less_than_or_equal)
    merge_sort_nb(array.copy(), is_less_than_or_equal)


    print("library sort")
    t = time.time()
    for i in range(1000):
        a = array.copy()
        a.sort()
    print(time.time() - t)

    print("insertion sort")
    t = time.time()
    for i in range(1000):
        a = array.copy()
        insertion_sort_nb(a, is_less)
    print(time.time() - t)

    print("quick sort iterative")
    t = time.time()
    for i in range(1000):
        a = array.copy()
        quick_sort_iterative_nb(a, is_less_than_or_equal)
    print(time.time() - t)

    print("quick sort recursive")
    t = time.time()
    for i in range(1000):
        a = array.copy()
        quick_sort_recursive_nb(a, is_less_than_or_equal)
    print(time.time() - t)

    print("merge sort")
    t = time.time()
    for i in range(1000):
        a = array.copy()
        merge_sort_nb(a, is_less_than_or_equal)
    print(time.time() - t)


