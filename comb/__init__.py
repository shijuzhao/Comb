"""Comb: a plug-and-play caching system for long-context LLM serving"""

import typing

MODULE_ATTRS = {
    "COMB": ".entrypoints.comb:COMB"
}

if typing.TYPE_CHECKING:
    from comb.entrypoints.comb import COMB
else:
    def __getattr__(name: str) -> typing.Any:
        from importlib import import_module

        if name in MODULE_ATTRS:
            module_name, attr_name = MODULE_ATTRS[name].split(":")
            module = import_module(module_name, __package__)
            return getattr(module, attr_name)
        else:
            raise AttributeError(
                f'module {__package__} has no attribute {name}')
        
__all__ = [
    "COMB"
]