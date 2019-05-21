from PIL import Image
import pandas
import numpy as np

im = Image.open('D:\Masters\Projects\\520\Machine Learning\DataSet\car2.jpg','r')  # relative path to file
# #load the pixel info
pix = im.load()
width, height = im.size

# read the details of each pixel and write them to the file
data = []
for x in range(width):
    dataCol = []
    for y in range(height):
          r = pix[x, y][0]
          g = pix[x, y][1]
          b = pix[x, y][2]
          # Gray(r, g, b) = 0.21r + 0.72g + 0.07b
          y = 0.21*(r) + 0.72*(g) + 0.07*(b)
          dataCol = [r,g,b,y]
          data.append(dataCol)
data = pandas.DataFrame(data)
data.to_csv('D:\Masters\Projects\\520\Machine Learning\DataSet\image2.csv')

# data = pandas.DataFrame(data,columns=['r','g','b','grey'])
# input = data['grey']
#
# outputData = data.drop(['grey'],axis=1)
#
# count = 0
# output = []
# for i in outputData:
#     if count%4 == 0:
#         output.append(i)
#     count+=1
#
# output = pandas.DataFrame(output,columns=['grey'])
#
# print(output)
