class Registry(object):

    def __init__(self, name=None):

        if name == None:
            self._name = "Registry"
        self._name = name
        self._obj_dict = dict()

    def __registry(self, obj, name=None, **kwargs):
        if name is None:
            name = obj.__name__
        self._obj_dict[name] = obj

    def registry(self, obj=None, name=None):

        if obj == None:
            def _no_obj_registry(func__or__class):
                self.__registry(func__or__class, name)
                return func__or__class
                                                
            return _no_obj_registry

        self.__registry(obj, name)

    def get(self, name):
        assert (name in self._obj_dict.keys()), f"{name} not in {self._name}!"
        return self._obj_dict[name]

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._obj_dict})'
        return format_str

MODELS = Registry("BACKBONES")
IMAGE_TRANSFORM = Registry("IMAGE_TRANSFORM")
LOSSES = Registry("LOSSES")
