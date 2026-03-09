import json
import os
import re
import threading
from pathlib import Path

from flask import Flask, jsonify, request

from biomni.agent import A1
from bioset_biomni.prompt import DEFAULT_DATASET, build_prompt

app = Flask(__name__)

_agent: A1 | None = None
_agent_lock = threading.Lock()

# Max steps per mode for the label endpoint
_MODE_MAX_STEPS = {
    "minimal": 5,
    "db": 15,
    "full": 30,
}

# HGNC dataset config
_HGNC_GDRIVE_ID = "1znogniT4GLa_HieLXoE8mO42TA6-UAd_"
_HGNC_FILENAME = "hgnc_complete_set.tsv"
_HGNC_DESCRIPTION = (
    "HUGO Gene Nomenclature Committee (HGNC) complete gene set. "
    "Contains unique symbols and names for human loci, including protein coding genes, "
    "ncRNA genes and pseudogenes. Can give canonical names for biomarker names in CyCIF. "
    "You can use it if you encounter a gene you do not know about, to get its aliases and official names."
)
_DATA_DIR = Path(__file__).parent / "data"


def _ensure_hgnc(agent: A1) -> None:
    """Download the HGNC dataset if absent and register it with the agent."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = _DATA_DIR / _HGNC_FILENAME

    if not dest.exists():
        print(f"[HGNC] Downloading to {dest} ...")
        try:
            import gdown
            gdown.download(id=_HGNC_GDRIVE_ID, output=str(dest), quiet=False)
        except Exception as e:
            print(f"[HGNC] Download failed: {e}")
            return

    agent.add_data({str(dest): _HGNC_DESCRIPTION})


def _parse_solution(text: str) -> dict:
    """Extract a JSON object from a <solution> block or bare text.

    Looks for <solution>...</solution> first, then finds the outermost
    { ... } within it and parses as JSON.
    """
    sol_match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    content = sol_match.group(1).strip() if sol_match else text.strip()

    # Find the outermost JSON object
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in solution. Raw output: {content[:300]}")

    return json.loads(json_match.group(0))


@app.post("/init")
def init():
    """Initialise the A1 agent.

    Body (JSON, all optional):
        llm     : model id (default: claude-sonnet-4-6)
        mode    : "full" | "db" | "minimal"  (default: "full")
        dataset : dataset description used to specialise the prompt
                  (default: "melanoma CyCIF")
        api_key : override API key
    """
    data = request.get_json(force=True, silent=True) or {}

    llm = data.get("llm", "claude-sonnet-4-6")
    mode = data.get("mode", "full")
    dataset = data.get("dataset", DEFAULT_DATASET)
    api_key = data.get("api_key") or None

    global _agent
    with _agent_lock:
        try:
            kwargs = dict(llm=llm, mode=mode, custom_prompt=build_prompt(dataset))
            if api_key:
                kwargs["api_key"] = api_key
            _agent = A1(**kwargs)
            _ensure_hgnc(_agent)
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


@app.post("/label")
def label():
    """Run the agent on a labelling task.

    Body (JSON):
        labelling : dict   – the labelling specification (required)
        mode      : str    – "full" | "db" | "minimal"; controls max_steps
        image     : str    – base64-encoded image (optional)

    Returns the parsed JSON object from the agent's <solution> block.
    """
    global _agent

    with _agent_lock:
        if _agent is None:
            return jsonify({"error": "Agent not initialised. Call POST /init first."}), 400

    data = request.get_json(force=True, silent=True) or {}

    labelling = data.get("labelling")
    if labelling is None:
        return jsonify({"error": "Missing required field: 'labelling'"}), 400

    mode = data.get("mode", "full")
    image_b64 = data.get("image") or None
    max_steps = _MODE_MAX_STEPS.get(mode, _MODE_MAX_STEPS["full"])

    prompt = (
        "You are given the following labelling task. "
        "Analyse the provided data and return your answer ONLY as a JSON object "
        "inside a <solution> tag — no other text outside the tag.\n\n"
        f"Labelling specification:\n{json.dumps(labelling, indent=2)}"
    )

    try:
        with _agent_lock:
            _, response = _agent.go(prompt, image=image_b64, max_steps=max_steps)
        result = _parse_solution(response)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def start_server(port: int = 5000, debug: bool = False):
    """Start the Biomni Flask server.

    Args:
        port  : TCP port to listen on (default 5000).
        debug : Enable Flask debug mode (default False).
    """
    app.run(host="0.0.0.0", port=port, debug=debug)
