"""Safe Python code execution environment for data calculation and visualization."""

import io
import sys
import math
import datetime
import json
import traceback
from typing import Any, Dict

# Pre-load optional libraries once at module initialization
_OPTIONAL_MODULES = {}
for _m in ["pandas", "numpy"]:
    try:
        _OPTIONAL_MODULES[_m] = __import__(_m)
    except Exception:
        pass


def execute_python_code(code: str, timeout_seconds: int = 15) -> Dict[str, Any]:
    """
    Executes Python code in a controlled namespace and returns captured stdout,
    generated variables, or error traces.
    """
    # Clean markdown formatting if passed by model
    clean_code = code.strip()
    if clean_code.startswith("```python"):
        clean_code = clean_code[9:]
    elif clean_code.startswith("```"):
        clean_code = clean_code[3:]
    if clean_code.endswith("```"):
        clean_code = clean_code[:-3]
    clean_code = clean_code.strip()

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout

    # Restricted built-in environment
    safe_globals = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
        }
    }

    safe_globals["math"] = math
    safe_globals["datetime"] = datetime
    safe_globals["json"] = json
    for mod_name, mod_obj in _OPTIONAL_MODULES.items():
        safe_globals[mod_name] = mod_obj

    local_vars: Dict[str, Any] = {}

    try:
        sys.stdout = stdout_capture
        # Execute code
        exec(clean_code, safe_globals, local_vars)
        output = stdout_capture.getvalue()
        
        # Format any interesting local variables
        exportable = {}
        for k, v in local_vars.items():
            if not k.startswith("_"):
                try:
                    exportable[k] = str(v)[:500]
                except Exception:
                    pass

        return {
            "success": True,
            "stdout": output.strip(),
            "variables": exportable,
            "error": None,
        }
    except Exception:
        return {
            "success": False,
            "stdout": stdout_capture.getvalue().strip(),
            "variables": {},
            "error": traceback.format_exc(),
        }
    finally:
        sys.stdout = old_stdout
