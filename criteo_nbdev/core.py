# AUTOGENERATED! DO NOT EDIT! File to edit: 00_core.ipynb (unless otherwise specified).

__all__ = ['StopExecution', 'skip', 'run_all']

# Cell
run_all = False

class StopExecution(Exception):
    def _render_traceback_(self):
        pass

def skip():
    if not run_all:
        raise StopExecution