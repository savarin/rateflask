import dill as pickle


def dump_to_pickle(file_object, file_path):
    pickle.dump(file_object, open(file_path, 'w'))


def load_from_pickle(file_path):
    return pickle.load(open(file_path, 'r'))