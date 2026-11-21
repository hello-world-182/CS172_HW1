"""Execute main.ipynb in place with a fresh kernel using jupyter_client."""
import json
import sys
from pathlib import Path

from jupyter_client.manager import start_new_kernel

path = Path(sys.argv[1] if len(sys.argv) > 1 else "main.ipynb")
kernel_name = sys.argv[2] if len(sys.argv) > 2 else "python3"
nb = json.loads(path.read_text(encoding="utf-8"))
km, kc = start_new_kernel(kernel_name=kernel_name, cwd=str(path.parent.resolve()))
count = 0
failed = None
try:
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs = []
        msg_id = kc.execute("".join(cell["source"]))
        while True:
            msg = kc.get_iopub_msg(timeout=1800)
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "status" and content["execution_state"] == "idle":
                break
            if kind == "stream":
                if outputs and outputs[-1]["output_type"] == "stream" and outputs[-1]["name"] == content["name"]:
                    outputs[-1]["text"].append(content["text"])
                else:
                    outputs.append({"output_type": "stream", "name": content["name"], "text": [content["text"]]})
            elif kind in ("display_data", "execute_result"):
                out = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                if kind == "execute_result":
                    out["execution_count"] = count
                outputs.append(out)
            elif kind == "error":
                outputs.append({"output_type": "error", "ename": content["ename"], "evalue": content["evalue"],
                                "traceback": content["traceback"]})
                failed = (index, content["ename"], content["evalue"])
        for out in outputs:
            if out["output_type"] == "stream":
                out["text"] = "".join(out["text"]).splitlines(keepends=True)
        cell["outputs"] = outputs
        cell["execution_count"] = count
        status = "ERROR" if (failed and failed[0] == index) else "ok"
        print(f"cell {index}: {status}", flush=True)
        if failed:
            break
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
if failed:
    print("FAILED", failed)
    sys.exit(1)
print("all cells executed")
