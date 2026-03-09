"""Entry point for starting the Biomni server.

Usage (from the project root or from a parent project that includes
Biomni as a git submodule):

    python biomni/run_server.py --port 5000
    python biomni/run_server.py --port 8080 --debug

Or from Python:

    from biomni.bioset_biomni import start_server
    start_server(port=5000)
"""

import argparse

from bioset_biomni import start_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the Biomni Flask server.")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    start_server(port=args.port, debug=args.debug)
