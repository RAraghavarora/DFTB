from tensorflow.python.client import device_lib
local_device_protos = device_lib.list_local_devices()
print(local_device_protos)