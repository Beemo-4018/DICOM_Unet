
import numpy as np
import matplotlib.pyplot as plt

from pydicom import dcmread

path = "/Users/admin/Downloads/dicom/PAT034/D0001.dcm"
# print("Reading DICOM file from path: ", path)

x = dcmread(path) ### ' FileDataset'\

# print(dir(x))
print(x.PixelSpacing)
print(x.pixel_array)

Dicom_file = x
plt.imshow(Dicom_file.pixel_array, cmap=plt.cm.gray)
plt.show()