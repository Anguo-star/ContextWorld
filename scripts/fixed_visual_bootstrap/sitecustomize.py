"""Bootstrap native LeWM dataset workers and DDP children for this control."""
import importlib
import os
import sys

if os.environ.get("CW_FIXED_VISUAL_BOOTSTRAP") == "1":
    try:
        try:
            importlib.import_module("flash_attn.modules.mha")
        except ModuleNotFoundError:
            pass
        except (ImportError, OSError):
            for name in tuple(sys.modules):
                if name == "flash_attn" or name.startswith("flash_attn."):
                    sys.modules.pop(name, None)
            sys.modules["flash_attn"] = None
        from contextworld.training.stablewm_bundle import register_stablewm_bundle_format
        register_stablewm_bundle_format()
    except BaseException as exc:
        sys.stderr.write(f"fixed visual bootstrap failed: {exc}\n")
        raise SystemExit(2) from exc
