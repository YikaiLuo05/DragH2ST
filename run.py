"""Usage: python run.py prepare|train|evaluate|serve|build-library|search."""
import os
import sys

# The reference's DLPFC barcode intersection uses Python sets. Establish the
# reproducible process hash seed before imports; this is not a model option.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def main():
    actions = {"prepare", "train", "evaluate", "serve", "build-library", "search"}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit(__doc__)
    from drag_h2st.config import load_config
    cfg = load_config()
    action = sys.argv[1]
    if action == "serve":
        from drag_h2st.prototype_bank.server import serve
        serve(cfg)
    elif action == "search":
        from drag_h2st.search import search
        search(cfg)
    else:
        from drag_h2st.pipeline import prepare, train, evaluate, build_library
        {"prepare": prepare, "train": train, "evaluate": evaluate,
         "build-library": build_library}[action](cfg)


if __name__ == "__main__":
    main()
