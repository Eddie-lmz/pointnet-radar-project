import os
filePath = r"C:\Users\86159\PycharmProjects\lidar\Pointnet_Pointnet2_pytorch-master\data\MyData\Cyclinder"
#####最后一行会出现一个， 报错！！！！！！！！！
######手动删除或改进程序
#####
file = '1.txt'
with open(file,'a') as f:
    f.write("[")
    for i,j,k in os.walk(filePath):
         for name in k:
              base_name=os.path.splitext(name)[0]  #去掉后缀 .txt
              f.write(" \"")
              f.write(os.path.join("shape_data/Cyclinder/",base_name))
              f.write("\"")
              f.write(",")
    f.write("]")
f.close()