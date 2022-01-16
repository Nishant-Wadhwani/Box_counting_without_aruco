from parser import vision
input_dir = '/home/nishant/Wipro/work/Dataset/data40'
import glob
img_list = glob.glob(input_dir+'/*')
for img_name in img_list:
    #print(img_name)
    L = vision(img_name)
    print(L)