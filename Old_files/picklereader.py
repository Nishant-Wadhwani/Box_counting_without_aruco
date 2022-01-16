import pickle


with open('box_count/tools/Dictionary/WCCL_SKU_db.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)