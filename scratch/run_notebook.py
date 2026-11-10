"""Execute the notebook cell by cell with timing, so slow cells are visible immediately."""

import sys
import time

import nbformat
from nbclient import NotebookClient

PATH = sys.argv[1] if len(sys.argv) > 1 else "PPI_Link_Prediction_SNA.ipynb"
nb = nbformat.read(PATH, as_version=4)

started = {}
t0 = time.time()


def on_cell_start(cell, cell_index):
    started["t"] = time.time()
    head = cell.source.strip().split("\n")[0][:70]
    print(f"[{time.time() - t0:7.1f}s] >>> cell {cell_index:2d} | {head}", flush=True)


def on_cell_complete(cell, cell_index):
    dur = time.time() - started.get("t", time.time())
    print(f"[{time.time() - t0:7.1f}s] <<< cell {cell_index:2d} took {dur:6.1f}s", flush=True)


def on_cell_error(cell, cell_index, execute_reply):
    print(f"!!! cell {cell_index} ERRORED", flush=True)


client = NotebookClient(
    nb,
    timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 600,
    kernel_name="python3",
    allow_errors=True,
    record_timing=True,
)
client.on_cell_start = on_cell_start
client.on_cell_complete = on_cell_complete
client.on_cell_error = on_cell_error

print("executing ...", flush=True)
client.execute()
nbformat.write(nb, PATH)
print(f"done in {time.time() - t0:.1f}s -> {PATH}", flush=True)
