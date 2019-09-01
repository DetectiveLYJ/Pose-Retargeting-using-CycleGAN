import os
import shutil

# compare_dir= '../../Data/compare_data/'

part='../Data/compare_data256/out21-frames_part1/'
all='../Data/compare_data256/out21/'

bias=2810

names= os.listdir(part)
for name in names:
    old_name=os.path.join(part,name)
    new_name=os.path.join(part,str(int(name[:-4])+bias)+'.png')
    print(new_name)
    os.rename(old_name, new_name)
    if not os.path.exists(all):
        os.makedirs(all)
    shutil.copy(new_name, all)


