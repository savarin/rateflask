import pickle


def dump_to_pickle(file_object, file_path):
    pickle.dump(file_object, open(file_path, 'w'))