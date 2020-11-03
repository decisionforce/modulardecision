from enum import Enum
class RoadOption(Enum):
    VOID = -1
    LEFT = 1 
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4

# file1 = open("/mnt/lustre/sunjiankai/lustrenew/CARLA/log/stochastic_easy_bak.pkl", "rb")
file1 = open("/data/Program/log/Overtake.pkl", "rb")
import pickle
def read_file(_file):
    data = []
    nums = 0
    while True:
        try:
            # import pdb; pdb.set_trace()
            data.append(pickle.load(_file))
            nums += 1
        except:
            break
    print(nums, "DATA NUMBER")
    return data, nums
def save_file(datas):
    # save_file = open("/mnt/lustre/sunjiankai/lustrenew/CARLA/log/stochastic_easy_50.pkl", "wb")
    save_file = open("/data/Program/log/Overtake_50.pkl", "wb")
    i = 0
    data_save = []
    for data in datas:
        for item in data:
            data_save.append(item)
            i += 1
            print(len(item))
    print("LOAD FINISHED", i)
    j = -1
    for data in data_save:
        j += 1
        if j % 200 >= 50:
            continue
        pickle.dump(data, save_file)    
    print("SAVE FINISHED", j)
    save_file.close()
    return 


if __name__ == '__main__':
    data1, nums1 = read_file(file1)
    data = [data1,]
    save_file(data)
    print("STATISTICS: ", nums1)
    print("Finished")
