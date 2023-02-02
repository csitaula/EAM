# dataset preparation
# for AID
# 5:5 and 2:8 for train/test, design 5 sets with different random number
path="//ad.monash.edu/home/User066/csit0004/Desktop/Jagannath_dai/AID/"
import splitfolders
seed=[100,200,300,400,500]
dir="//ad.monash.edu/home/User066/csit0004/Desktop/Jagannath_dai/AID_/2_8/"
path_subdir=["1","2","3","4","5"]
for s in range(0, len(seed)):
    splitfolders.ratio(input=path,output=dir+path_subdir[s],seed=seed[s],ratio=(0.2,0.8),group_prefix=None)
    print('Finished for set:'+str(s+1))



# for NWPU
# for 2:8 and 1:9 for train/test set, design 5 sets with different random number
path="//ad.monash.edu/home/User066/csit0004/Desktop/Jagannath_dai/NWPU-RESISC45/"
seed=[100,200,300,400,500]
dir="//ad.monash.edu/home/User066/csit0004/Desktop/Jagannath_dai/NWPU_/1_9/"
path_subdir=["1","2","3","4","5"]
for s in range(0, len(seed)):
    splitfolders.ratio(path,output=dir+path_subdir[s],seed=seed[s],ratio=(0.1,0.9),group_prefix=None)
    print('Finished for set:' + str(s + 1))
