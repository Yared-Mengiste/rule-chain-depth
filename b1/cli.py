from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .io import read_worlds, write_json
from .oracle import validate
from .pilot import handmade_pilot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B1 rule-chain depth benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle", help="Cross-check both solvers; stop on any disagreement")
    oracle.add_argument("--worlds", type=Path, default=Path("data/pilot/worlds.json"))
    oracle.add_argument("--expected", type=Path, default=Path("data/pilot/hand_expected.json"))
    oracle.add_argument("--output", type=Path, default=Path("artifacts/oracle/pilot.answer_key.json"))
    author = commands.add_parser("author-pilot", help="Freeze the eight hand-authored worlds and annotations")
    author.add_argument("--output-dir", type=Path, default=Path("data/pilot"))
    full = commands.add_parser("author-benchmark", help="Freeze and validate the balanced 40-question dataset")
    full.add_argument("--output-dir", type=Path, default=Path("data/benchmark"))
    benchmark = commands.add_parser("benchmark", help="Compare PLN-RAG, normal RAG, and full context on identical questions")
    benchmark.add_argument("--systems", nargs="+", choices=("pln_rag", "rag", "full_context"),
                           default=["pln_rag", "rag", "full_context"])
    benchmark.add_argument("--dataset", choices=("benchmark", "pilot"), default="benchmark")
    benchmark.add_argument("--worlds", type=Path)
    benchmark.add_argument("--expected", type=Path)
    benchmark.add_argument("--world-limit", type=int, help="Select the first N worlds; omitted means the full dataset")
    benchmark.add_argument("--pln-runtime", choices=("docker", "local"), default="docker")
    benchmark.add_argument("--container", help="Existing PLN-RAG container used to select its image")
    benchmark.add_argument("--repo", type=Path, default=Path("../PLN-RAG"))
    benchmark.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
    benchmark.add_argument("--normal-rag-repo", type=Path, default=Path("../normal-rag"))
    benchmark.add_argument("--qdrant-url")
    benchmark.add_argument("--ollama-url")
    benchmark.add_argument("--run-id")
    benchmark.add_argument("--prepare-only", action="store_true", help="Validate and save a plan without model or embedding calls")
    for name, help_text in (("doctor", "Check the live runtime without calling the LLM"),
                            ("pilot", "Run only the eight-world PLN-RAG pilot; save all raw outputs")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--repo", type=Path, default=Path("../PLN-RAG"))
        command.add_argument("--python", type=Path, default=Path(".venv/bin/python"))
        if name == "doctor":
            command.add_argument("--output", type=Path, default=Path("artifacts/setup/doctor.json"))
        else:
            command.add_argument("--worlds", type=Path, default=Path("data/pilot/worlds.json"))
            command.add_argument("--expected", type=Path, default=Path("data/pilot/hand_expected.json"))
            command.add_argument("--run-id")
            command.add_argument("--world-limit", type=int, default=8,
                                 help="Run the first N validated worlds; N<8 is labelled a trial")
    docker = commands.add_parser("docker-pilot", help="Run a private trial from a running PLN-RAG container's image")
    docker.add_argument("--container", required=True, help="Existing PLN-RAG container name")
    docker.add_argument("--world-limit", type=int, default=2)
    docker.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    docker.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/embeddings")
    docker.add_argument("--prepare-only", action="store_true", help="Inspect and save the launch plan without running it")
    args = parser.parse_args(argv)
    try:
        if args.command in ("author-pilot", "author-benchmark"):
            if args.command == "author-benchmark":
                from .dataset import full_dataset, validate_full
                worlds, expected = full_dataset()
                validate_full(worlds, expected)
            else:
                worlds, expected = handmade_pilot()
            # No fixture is admitted without the independent checks.
            validate(worlds, expected)
            write_json(args.output_dir / "worlds.json", {"schema_version": 1, "worlds": [w.to_dict() for w in worlds]})
            write_json(args.output_dir / "hand_expected.json", expected)
            print(f"Authored and validated {len(worlds)} worlds / {len(expected)} questions in {args.output_dir}")
        elif args.command == "benchmark":
            from .comparison import run_comparison
            output = run_comparison(Path.cwd(), systems=args.systems, dataset_kind=args.dataset,
                worlds_path=args.worlds, expected_path=args.expected, world_limit=args.world_limit,
                normal_repo=args.normal_rag_repo, pln_runtime=args.pln_runtime, container=args.container,
                repo=args.repo, python=args.python, qdrant_url=args.qdrant_url, ollama_url=args.ollama_url,
                prepare_only=args.prepare_only, run_id=args.run_id)
            if args.prepare_only:
                return 0
            summary = json.loads((output / "summary.json").read_text())
            print(f"Status: {summary['status']}")
            return 0 if summary["status"] == "COMPLETE" else 2
        elif args.command == "oracle":
            worlds = read_worlds(args.worlds)
            expected = json.loads(args.expected.read_text(encoding="utf-8"))
            result = validate(worlds, expected)
            write_json(args.output, result)
            print(f"AGREED: {result['world_count']} worlds, {result['question_count']} questions; {args.output}")
        elif args.command == "doctor":
            from .doctor import doctor
            result = doctor(Path.cwd(), args.repo.resolve(), args.python.absolute(), args.output)
            print(json.dumps({k: v for k, v in result.items() if k != "runtime_log"}, indent=2))
            print(f"Setup details: {args.output}")
            return 0 if result["ready"] else 2
        elif args.command == "pilot":
            from .pln_runner import run_pilot
            output = run_pilot(Path.cwd(), args.repo.resolve(), args.python.absolute(),
                               args.worlds, args.expected, args.run_id, args.world_limit)
            print(f"Raw pilot outputs: {output / 'raw.md'}")
            gate = json.loads((output / "gate.json").read_text())
            print(f"Gate: {gate['status']}")
            return 0 if gate["status"] in ("PILOT_PASSED", "TRIAL_COMPLETE") else 2
        elif args.command == "docker-pilot":
            from .docker_runtime import docker_pilot
            return docker_pilot(Path.cwd(), args.container, args.world_limit,
                                args.qdrant_url, args.ollama_url, args.prepare_only)
        return 0
    except Exception as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
