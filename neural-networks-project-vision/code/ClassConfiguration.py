class ClassConfiguration:
    """Represents the configuration of an arbitrary class with its type and arguments

    Parameters
    ----------
    class_type : type
        The type of the class
    kwargs : dict
        Dictionary of keyword arguments to pass to the class' constructor

    Attributes
    ----------
    class_type : type
        The type of the class
    kwargs : dict
        Dictionary of keyword arguments to pass to the class' constructor
    """
    def __init__(self, class_type, kwargs):
        self.class_type = class_type
        self.kwargs = kwargs
        
    def instance(self, *args):
        """Creates an instance of the represented class with the provided positional and keyword arguments
        """
        return self.class_type(*args, **self.kwargs)
