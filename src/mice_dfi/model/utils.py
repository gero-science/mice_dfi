import os


def make_rec_dir(name):
    """ Creates directory with `name`, if directory exist add integer suffix at the end  """
    def get_name_rec_dir(name_):
        n = 0
        suffix = ''
        name_ = os.path.relpath(name_)
        while os.path.exists(name_ + suffix):
            suffix = '_%d' % n
            n += 1
        return name_ + suffix
    name = get_name_rec_dir(name)
    os.makedirs(name)
    return name
