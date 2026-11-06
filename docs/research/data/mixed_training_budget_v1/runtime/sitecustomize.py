"""Isolate broken optional flash-attn binary so Kornia uses its documented fallback."""
import builtins
_real_import = builtins.__import__
def _optional_flash_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'flash_attn' or name.startswith('flash_attn.'):
        raise ModuleNotFoundError("Optional flash_attn disabled: binary incompatible with local torch", name='flash_attn')
    return _real_import(name, globals, locals, fromlist, level)
builtins.__import__ = _optional_flash_import
