"""Restricted pickle loader: refuses to import any class or function, so a pickle can only
produce plain Python data (dict, list, tuple, str, int, float, bool, None). Used for the
Recipe1MSubs files, which we verified contain no object-creating opcodes."""
import io
import pickle


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"blocked attempt to load {module}.{name}")


def load(path):
    with open(path, "rb") as fh:
        return _DataOnlyUnpickler(io.BufferedReader(fh)).load()
