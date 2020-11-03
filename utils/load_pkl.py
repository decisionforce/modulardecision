import pickle

f1 = open('deterministic_trpo_overtake.pkl', 'rb')
data1 = pickle.load(f1)

print('data1: ', data1)

f2 = open('OtherLeadingVehicle_TRPO_curriculum.pkl', 'rb')
data2 = pickle.load(f2)

print('data2: ', data2)

for i in range(10):
    data1 = pickle.load(f1)
    data2 = pickle.load(f2)

    print(i, 'data1: ', data1)
    print(i, 'data2: ', data2)