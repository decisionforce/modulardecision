from enum import Enum
class RoadOption(Enum):
    VOID = -1
    LEFT = 1 
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4

_file = open("NEW_OVERTAKE.pkl", "rb")
import pickle
def read_file(_file):
    data = []
    nums = 0
    while True:
        try:
            item = pickle.load(_file)
            data.append(item)
            nums += 1
        except:
            break
    print(nums, "DATA NUMBER")
    return data, nums

if __name__ == '__main__':
    data, nums = read_file(_file)
    print(nums)
    print(len(data))
    print("Finished")
