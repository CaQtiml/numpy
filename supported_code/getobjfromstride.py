import numpy as np
import ctypes

class A: pass

a = A()
b = A()
c = A()

print(f"Creating array with objects: {a}, {b}, {c}")
arr = np.array([a, b, c], dtype=object)

# Get the raw char* pointer (as integer address)
data_ptr = arr.ctypes.data  # base address of buffer
itemsize  = arr.itemsize    # 8 bytes (pointer size)

print(f"Buffer address : {hex(data_ptr)}")
print(f"Item size      : {itemsize} bytes")
print()

for i in range(len(arr)):
    # Compute address of each pointer slot
    slot_address = data_ptr + i * itemsize

    # Read the PyObject* stored at that slot
    pyobj_address = ctypes.cast(
        slot_address,
        ctypes.POINTER(ctypes.c_size_t)  # read 8-byte pointer
    ).contents.value

    # Cast the address back to a Python object
    obj = ctypes.cast(pyobj_address, ctypes.py_object).value

    print(f"[{i}] slot={hex(slot_address)}  obj_ptr={hex(pyobj_address)}  value={obj!r}")

# The result is
# Creating array with objects: <__main__.A object at 0x7fff74ac9d20>, <__main__.A object at 0x7fff74974480>, <__main__.A object at 0x7fff74974740>
# Buffer address : 0x55555600a9d0
# Item size      : 8 bytes

# [0] slot=0x55555600a9d0  obj_ptr=0x7fff74ac9d20  value=<__main__.A object at 0x7fff74ac9d20>
# [1] slot=0x55555600a9d8  obj_ptr=0x7fff74974480  value=<__main__.A object at 0x7fff74974480>
# [2] slot=0x55555600a9e0  obj_ptr=0x7fff74974740  value=<__main__.A object at 0x7fff74974740>