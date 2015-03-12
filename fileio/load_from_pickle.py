import pickle


def load_from_pickle(file_path):
    return pickle.load(open(file_path, 'r'))