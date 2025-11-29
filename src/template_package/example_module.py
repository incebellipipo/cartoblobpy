#!/usr/bin/env python3

from __future__ import annotations

def template_function(i: int = 0) -> str:
    """
    This is where you place the doc string for your function.

    :param i: An integer
    :type i: int
    :return: A string
    :rtype: str
    """
    return f"string-{i}"

class TemplateClass:
    """
    A template class.

    This is where you place the doc string for your class.
    """
    def __init__(self, i: int = 0):
        """
        Doc String for TemplateClass.

        :param i: An integer
        :type i: int
        """
        self.i = i

    def template_method(self) -> str:
        """
        Doc String for template_method.

        :return: A string
        :rtype: str
        """
        return f"string-{self.i}"

    def simple_method(self, a: float = 0, b: float = 0) -> float:
        r"""
        Doc String for simple_method.

        'r' is necessary for the math to render properly. This makes a multiline
        string a raw string.

        .. math::

            x' = a + b


        :param a: A float, Here you need to describe what `a` is. What is it
            used for? What does it represent? You can expand on this in the
            description.
        :param b: Another float
        :retuns: a + b
        :rtype: float
        """
        return a + b

    @staticmethod
    def template_static_mathematical_method(i: int, j: int) -> int:
        r"""
        Doc String for template_static_mathematical_method.

        You can include latex too!

        .. math::

            x' = i + j

        This is where you place examples of how to use your method.

        .. code-block:: python

            from template_package import example_module
            example_module.template_static_mathematical_method(1, 2)

        :param i: An integer, Here you need to describe what `i` is. What is it
            used for? What does it represent? You can expand on this in the
            description.
        :type i: int

        :param j: Another integer
        :type j: int

        :return: i + 1
        :rtype: int
        """

        return i + j

