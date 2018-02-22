import pickle

def pklSave(data, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, 2)

def pklLoad(file_name):
    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    return(data)