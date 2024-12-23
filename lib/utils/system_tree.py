from jax.tree_util import register_pytree_node
from jax import tree_util
import mujoco
import os
from brax.io import mjcf
import jax

# class SystemsTree:
#     def __init__(self, systems):
#         self.systems = systems

#     def tree_flatten(self):
#         # Flatten the dictionary of systems into leaves and auxiliary keys
#         keys, values = zip(*self.systems.items())
#         return list(values), keys

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         # Reconstruct the systems dictionary from leaves and auxiliary keys
#         systems = {key: child for key, child in zip(aux_data, children)}
#         return cls(systems)

#     def __getitem__(self, key):
#         return self.systems[key]

#     def __setitem__(self, key, value):
#         self.systems[key] = value

#     def __len__(self):
#         return len(self.systems.keys())
    
#     def _get_item(self, key):
#         return self.systems[key]


from types import SimpleNamespace

class SystemsTree:
    def __init__(self, systems):
        # Convert the systems to a SimpleNamespace
        self.systems = self.dict_to_namespace(systems)

    def dict_to_namespace(self, d):
        """Recursively converts a dictionary to a namespace."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: self.dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [self.dict_to_namespace(item) for item in d]
        else:
            return d

    def tree_flatten(self):
        """Flatten the systems Namespace into leaves and auxiliary keys."""
        values = []
        keys = []

        def extract(namespace):
            # Recursively flatten the namespace into keys and values
            for key, value in namespace.__dict__.items():
                if isinstance(value, SimpleNamespace):
                    extract(value)  # Recursively extract
                else:
                    values.append(value)
                    keys.append(key)

        extract(self.systems)
        return values, keys

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct the systems Namespace from leaves and auxiliary keys."""
        systems = cls._rebuild_namespace(aux_data, children)
        return cls(systems)

    @staticmethod
    def _rebuild_namespace(aux_data, keys):
        """Helper function to rebuild namespace from flattened data."""
        systems = SimpleNamespace()
        for key, value in zip(keys, aux_data):
            setattr(systems, key, value)
        return systems

    def __getitem__(self, key):
        return self._get_item(self.systems, key)

    def _get_item(self, namespace, key):
        """Helper to recursively find the item in the namespace."""
        if hasattr(namespace, str(key)):
            return getattr(namespace, str(key))
        else:
            raise KeyError(f"Key '{str(key)}' not found in the namespace.")

    def __setitem__(self, key, value):
        self._set_item(self.systems, key, value)

    def _set_item(self, namespace, key, value):
        """Helper to recursively set the item in the namespace."""
        setattr(namespace, key, value)

    def __len__(self):
        return self._count_keys(self.systems)

    def _count_keys(self, namespace):
        """Helper function to count the number of keys in the namespace."""
        count = 0
        for _, value in namespace.__dict__.items():
            if isinstance(value, SimpleNamespace):
                count += self._count_keys(value)
            else:
                count += 1
        return count

# from collections import namedtuple
# from typing import Any, Dict

# # Define a namedtuple to represent a system with dynamic attributes
# System = namedtuple('System', ['name', 'value'])

# class SystemsTree:
#     def __init__(self, systems: Dict[str, Any]):
#         # Convert the systems dictionary to namedtuples
#         self.systems = self.dict_to_namedtuple(systems)

#     def dict_to_namedtuple(self, d: Dict[str, Any]):
#         """Recursively converts a dictionary to namedtuples."""
#         for key, value in d.items():
#             if isinstance(value, dict):
#                 d[key] = self.dict_to_namedtuple(value)
#             elif isinstance(value, list):
#                 d[key] = [self.dict_to_namedtuple(item) if isinstance(item, dict) else item for item in value]
#         return {key: namedtuple(key.capitalize(), value.keys())(*value.values()) if isinstance(value, dict) else value for key, value in d.items()}

#     def tree_flatten(self):
#         """Flatten the systems structure into leaves and auxiliary keys."""
#         values = []
#         keys = []

#         def extract(namespace):
#             """Recursively extract values and keys from namedtuples."""
#             for key, value in namespace._asdict().items():
#                 if isinstance(value, tuple):  # Check if it's a namedtuple
#                     extract(value)  # Recursively extract from nested namedtuple
#                 else:
#                     values.append(value)
#                     keys.append(key)

#         for system in self.systems.values():
#             extract(system)
#         return values, keys

#     @classmethod
#     def tree_unflatten(cls, aux_data, keys):
#         """Reconstruct the systems structure from flattened data."""
#         systems = cls._rebuild_namedtuple(aux_data, keys)
#         return cls(systems)

#     @staticmethod
#     def _rebuild_namedtuple(aux_data, keys):
#         """Helper function to rebuild a namedtuple from flattened data."""
#         systems = {}
#         for key, value in zip(keys, aux_data):
#             name, attributes = key.split("_", 1)
#             systems[key] = namedtuple(name, attributes.split("_"))(*value)
#         return systems

#     def __getitem__(self, key):
#         return self._get_item(self.systems, key)

#     def _get_item(self, systems, key):
#         """Helper to get an item from the systems."""
#         if key in systems:
#             return systems[key]
#         else:
#             raise KeyError(f"Key '{key}' not found in the systems.")

#     def __setitem__(self, key, value):
#         self._set_item(self.systems, key, value)

#     def _set_item(self, systems, key, value):
#         """Helper to set an item in the systems."""
#         systems[key] = value

#     def __len__(self):
#         return len(self.systems)



if __name__=='__main__':
    register_pytree_node(
    SystemsTree,
    SystemsTree.tree_flatten,
    SystemsTree.tree_unflatten
)