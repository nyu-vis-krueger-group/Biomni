import json
import os
import re
import threading
from pathlib import Path

from flask import Flask, jsonify, request

from biomni.agent import A1
from biomni.config import default_config
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
        llm     : model id for the main agent (default: claude-sonnet-4-6)
        db_llm  : model id for database queries (default: same as llm)
        mode    : "full" | "db" | "minimal"  (default: "full")
        dataset : dataset description used to specialise the prompt
                  (default: "melanoma CyCIF")
        api_key : override API key
    """
    data = request.get_json(force=True, silent=True) or {}

    llm = data.get("llm", "claude-sonnet-4-6")
    db_llm = data.get("db_llm") or llm
    mode = data.get("mode", "full")
    dataset = data.get("dataset", DEFAULT_DATASET)
    api_key = data.get("api_key") or None

    global _agent
    with _agent_lock:
        try:
            default_config.llm = db_llm
            kwargs = dict(llm=llm, mode=mode, custom_prompt=build_prompt(dataset))
            if api_key:
                kwargs["api_key"] = api_key
            _agent = A1(**kwargs)
            _ensure_hgnc(_agent)
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500


def _check_init():
    """Return an error response if the agent is not initialised, else None."""
    with _agent_lock:
        if _agent is None:
            return jsonify({"error": "Agent not initialised. Call POST /init first."}), 400
    return None


def _run(task_json: dict, image_b64: str | None, mode: str) -> tuple:
    """Build the prompt, run the agent, and return a (result_dict, None) or (None, error_response)."""
    max_steps = _MODE_MAX_STEPS.get(mode, _MODE_MAX_STEPS["full"])
    prompt = (
        "Return your answer ONLY as a JSON object inside a <solution> tag — "
        "no other text outside the tag.\n\n"
        f"{json.dumps(task_json, indent=2)}"
    )
    try:
        with _agent_lock:
            _, response = _agent.go(prompt, image=image_b64, max_steps=max_steps)
        return _parse_solution(response), None
    except ValueError as e:
        return None, (jsonify({"error": str(e)}), 422)
    except Exception as e:
        return None, (jsonify({"error": str(e)}), 500)


def _extract_common(data: dict) -> tuple[list | None, dict | None, str | None, str]:
    """Extract and validate fields shared by all task endpoints.

    Returns (markers, channel_stats, image_b64, mode).
    markers is None when missing so callers can return an error.
    """
    markers = data.get("markers") or None
    channel_stats = data.get("channel_stats") or None
    image_b64 = data.get("image") or None
    mode = data.get("mode", "full")
    return markers, channel_stats, image_b64, mode


@app.post("/label")
def label():
    """Generate biological labels for a set of markers.

    Body (JSON):
        markers       : list[str]  – e.g. ["SOX10:#FFFF00", "PRAME:#FF0000"]  (required)
        channel_stats : dict       – full channel statistics for the region      (required)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                 (optional)

    Returns: {"labels": {...}, "overall": [...]}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    markers, channel_stats, image_b64, mode = _extract_common(data)
    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400
    if not channel_stats:
        return jsonify({"error": "Missing required field: 'channel_stats'"}), 400

    task_json = {"task": "label", "markers": markers, "channel_stats": channel_stats}
    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


@app.post("/query")
def query():
    """Answer a free-form question about a set of markers.

    Body (JSON):
        markers       : list[str]  – e.g. ["SOX10:#FFFF00", "PRAME:#FF0000"]  (required)
        query         : str        – the question to answer                     (required)
        channel_stats : dict       – full channel statistics for the region     (required)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                 (optional)

    Returns: {"answer": "..."}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    markers, channel_stats, image_b64, mode = _extract_common(data)
    question = data.get("query") or None
    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400
    if not question:
        return jsonify({"error": "Missing required field: 'query'"}), 400
    if not channel_stats:
        return jsonify({"error": "Missing required field: 'channel_stats'"}), 400

    task_json = {"task": "query", "markers": markers, "query": question, "channel_stats": channel_stats}
    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


@app.post("/suggest")
def suggest():
    """Recommend additional channels to enable alongside the currently selected markers.

    Body (JSON):
        markers       : list[str]  – currently selected markers                 (required)
        channel_stats : dict       – full channel statistics for the region     (required)
        mode          : str        – "full" | "db" | "minimal"
        image         : str        – base64-encoded JPEG or PNG                 (optional)

    Returns: {"suggestions": [{"channel": str, "reason": str, "priority": str}, ...]}
    """
    err = _check_init()
    if err:
        return err

    data = request.get_json(force=True, silent=True) or {}
    markers, channel_stats, image_b64, mode = _extract_common(data)
    if not markers:
        return jsonify({"error": "Missing required field: 'markers'"}), 400
    if not channel_stats:
        return jsonify({"error": "Missing required field: 'channel_stats'"}), 400

    task_json = {"task": "suggest", "markers": markers, "channel_stats": channel_stats}
    result, err = _run(task_json, image_b64, mode)
    return err if err else jsonify(result)


def start_server(port: int = 5000, debug: bool = False):
    """Start the Biomni Flask server.

    Args:
        port  : TCP port to listen on (default 5000).
        debug : Enable Flask debug mode (default False).
    """
    app.run(host="0.0.0.0", port=port, debug=debug)
