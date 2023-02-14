# dataset preparation
# for AID
# 5:5 and 2:8 for train/test, design 5 sets with different random number
# path="D://Jagannath_dai/AID/"
# import splitfolders
# seed=[100,200,300,400,500,600,700,800,900,1000, 1100,1200,1300,1400,1500,1600,1700,1800,1900,2000] # for 10 folds
# dir="D://Jagannath_dai/AID_/5_5/"
# path_subdir=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
# for s in range(0, len(seed)):
#     splitfolders.ratio(input=path,output=dir+path_subdir[s],seed=seed[s],ratio=(0.5,0.5),group_prefix=None)
#     print('Finished for set:'+str(s+1))


# for NWPU
import splitfolders
# for 2:8 and 1:9 for train/test set, design 5 sets with different random number
path="D://Jagannath_dai/NWPU-RESISC45/"
seed=[100,200,300,400,500,600,700,800,900,1000, 1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]
dir="D://Jagannath_dai/NWPU_/2_8/"
path_subdir=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20"]
for s in range(0, len(seed)):
    splitfolders.ratio(path,output=dir+path_subdir[s],seed=seed[s],ratio=(0.2,0.8),group_prefix=None)
    print('Finished for set:' + str(s + 1))
