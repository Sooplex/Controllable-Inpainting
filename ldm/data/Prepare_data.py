import os

path = r'/home/pnp/T2I-Adapter-SD/datasets/canny/images'  # 文件路径

filenames = os.listdir(path)
filenames.sort(key=lambda x: int(x[1:-4]))    # 解决自动排序问题：按照多少维排序

f = open('/home/pnp/T2I-Adapter-SD/datasets/canny/canny_RS.txt', 'w')       # 打开234.txt文件，进行写入
for name in filenames:
    # name = name.split(".")[0]  # 去后缀名
    # print(name)                # 查看是否去后缀名成功
    f.write(name + '\n')       # 写入txt文件中
f.close()
