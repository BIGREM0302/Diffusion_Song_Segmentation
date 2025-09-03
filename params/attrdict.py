# AttrDict inhertis class dict
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        # pass *args and **kwargs to class dict's __init__
        super(AttrDict, self).__init__(*args, **kwargs)
        # the following line enables obj.a = obj[a], while obj is still a dictionary
        self.__dict__ = self

    def override(self, attrs):
        # if atts is a kind of dict
        if isinstance(attrs, dict):
            # self.__dict__ = self, self.update(**attrs)
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            # if input is list, then it's array
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self
