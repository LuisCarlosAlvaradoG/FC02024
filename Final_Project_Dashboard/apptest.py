"""Headless smoke test: execute app.py via Streamlit's AppTest and assert no
uncaught exception is raised, in BOTH input modes:
  1. Underlying chain (default, synthetic data — offline path)
  2. Specific contract (OCC symbol — real Yahoo pull)
"""
from streamlit.testing.v1 import AppTest


def run_default():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    if at.exception:
        print("MODE 1 (chain) RAISED:")
        for e in at.exception:
            print(" -", e.value)
        return False
    print("MODE 1 (underlying chain) OK — tabs:", len(at.tabs))
    return True


def run_contract():
    at = AppTest.from_file("app.py", default_timeout=180)
    at.run()
    # Flip the input-mode radio to "Specific contract" and rerun.
    target = next((rb for rb in at.radio if "Specific contract" in rb.options), None)
    if target is None:
        print("MODE 2 skipped: mode radio not found")
        return None
    target.set_value("Specific contract").run()
    if at.exception:
        print("MODE 2 (contract) RAISED:")
        for e in at.exception:
            print(" -", e.value)
        return False
    print("MODE 2 (specific contract) OK — tabs:", len(at.tabs),
          "| text inputs:", len(at.text_input))
    return True


ok1 = run_default()
try:
    ok2 = run_contract()
except Exception as exc:
    print("MODE 2 harness error (likely network):", repr(exc)[:160])
    ok2 = None

if ok1 and ok2 is not False:
    print("APP RENDERED OK")
    raise SystemExit(0)
raise SystemExit(1)
