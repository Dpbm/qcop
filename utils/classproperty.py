"""Holds Class Property class."""

from typing import Any, Callable

class ClassProperty:
    """
    A class used as decorator
    to access properties as a class member (Without instantiation).

    https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    """

    def __init__(self, func:Callable):
        self.fget = func

    def __get__(self, obj:Any, owner:Any):
        return self.fget(owner)
