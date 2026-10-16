import argparse
import json
from pathlib import Path

from .dataset import validate
from .runner import SYSTEMS, benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description="B2: 42 fixed EntailmentBank worlds, 25 sentences each; PLN-RAG, RAG and full context.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Validate the checked-in bundle offline; no downloads or credentials")
    run = commands.add_parser("benchmark", help="Run all selected systems and automatically score/report evidence and resources")
    run.add_argument("--prepare-only", action="store_true", help="Write only an offline plan and request estimate; no service calls")
    run.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    run.add_argument("--world-limit", type=int, default=42, help="Select the first N worlds after validating all 42")
    run.add_argument("--output-dir", type=Path, help="New run directory; existing directories are never overwritten")
    run.add_argument("--run-id", help="Run name under artifacts/b2 (default: timestamp plus random suffix)")
    run.add_argument("--top-k", type=int, default=4, help="Fixed normal-RAG sentence retrieval limit (default: 4)")
    run.add_argument("--pln-top-k", type=int, default=10, help="Fixed native PLN vector context limit (default: 10)")
    run.add_argument("--pln-runtime", choices=("docker", "local"), default="docker")
    run.add_argument("--container", default="pln-rag-pln-rag-1")
    run.add_argument("--repo", type=Path, help="Local PLN-RAG checkout (default: ../PLN-RAG)")
    run.add_argument("--python", type=Path, help="Local PLN Python interpreter; baselines use the launcher interpreter")
    run.add_argument("--normal-rag-repo", type=Path, help="Normal-RAG components (default: ../normal-rag)")
    run.add_argument("--qdrant-url")
    run.add_argument("--ollama-url")
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        if command == "validate":
            result = validate()
            print(json.dumps({"valid": True, "world_count": len(result["worlds"]), "depth_counts": result["depth_counts"],
                "bundle_sha256": result["bundle_sha256"],
                "selected_annotation_discrepancies": {r["source_id"]: r["discrepancies"] for r in result["references"] if r["discrepancies"]}}, indent=2))
            return 0
        out = benchmark(Path.cwd(), **args)
        manifest = json.loads((out / "manifest.json").read_text())
        print(f"{manifest['status']}: {out}")
        print(f"{manifest['world_count']} worlds, {manifest['record_count']} records; initial chat estimate {manifest['request_estimate']['chat_calls']}; configured cap {manifest['llm_settings']['max_calls'] or 'unlimited'}. No automatic cap increase.")
        return 0 if manifest["status"] in ("PREPARED", "COMPLETE") else 2
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"b2: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
