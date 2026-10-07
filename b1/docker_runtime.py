"""Use an existing PLN-RAG image in a separate container, never its application API."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .environment import configured_environment
from .io import write_json
from .llm_control import LLMSettings


def launch_command(root: Path, inspected: dict, run_id: str, world_limit: int,
                   qdrant_url: str, ollama_url: str, *, job_path: Path | None = None,
                   forwarded_env: tuple[str, ...] = ()) -> list[str]:
    if not inspected.get("State", {}).get("Running"):
        raise ValueError("The selected PLN-RAG container is not running")
    image = inspected.get("Image", "")
    if not image.startswith("sha256:"):
        raise ValueError("Docker did not report an immutable image ID")
    if "," in str(root):
        raise ValueError("The benchmark path cannot contain a comma in Docker mount syntax")
    owner = root.stat()
    command = ["docker", "run", "--rm", "--pull", "never", "--init",
               "--name", "b1-" + run_id, "--network", "host",
               "--user", f"{owner.st_uid}:{owner.st_gid}",
               "--mount", f"type=bind,src={root},dst={root}", "--workdir", str(root),
               "--env", "B1_USE_ROOTLESS_RUNTIME=false",
               "--env", "B1_DOCKER_IMAGE=" + image,
               "--env", "B1_QDRANT_URL=" + qdrant_url,
               "--env", "OLLAMA_URL=" + ollama_url,
               "--env", "PYTHONDONTWRITEBYTECODE=1",
               "--entrypoint", "python3"]
    # Match the live container's tuned parser artifacts, read-only. Do not attach
    # its atomspace directory, Qdrant volume, or any other application data mount.
    for mount in inspected.get("Mounts", []):
        destination = mount.get("Destination")
        if destination not in ("/app/data/simba_canonical_pln.json", "/app/data/simba_all.json"):
            continue
        source = mount.get("Source", "")
        if mount.get("Type") != "bind" or not Path(source).is_file() or "," in source:
            raise ValueError("A live parser artifact cannot be mounted read-only from its reported host path")
        command += ["--mount", f"type=bind,src={source},dst={destination},readonly"]
    for key in forwarded_env:
        command += ["--env", key]  # Docker reads the value from its process environment.
    command += ([image, "-m", "b1.docker_runtime", "--inside-benchmark", str(job_path)]
                if job_path is not None else
                [image, "-m", "b1.docker_runtime", "--inside", run_id, str(world_limit)])
    return command


def prepare_benchmark(root: Path, container: str, job: dict, environment: dict) -> dict:
    """Inspect now and save the exact launch; credentials are never job fields."""
    inspection = subprocess.run(["docker", "inspect", "--type", "container", container],
                                capture_output=True, text=True, check=False)
    if inspection.returncode:
        raise RuntimeError("Cannot inspect PLN-RAG. Run with Docker access (sudo if required).")
    inspected = json.loads(inspection.stdout)[0]
    job_path = Path(job["output_dir"]).parent / "pln_job.json"
    write_json(job_path, job)
    names = ("OPENAI_API_KEY", "OPENAI_MODEL", "PARSER", "OPENROUTER_PROVIDER", "OPENROUTER_BASE_URL",
             "OLLAMA_MODEL", "B1_LLM_TEMPERATURE", "B1_LLM_MAX_TOKENS", "B1_MAX_LLM_CALLS",
             "B1_LLM_MIN_INTERVAL", "B1_LLM_TIMEOUT", "B1_QUERY_TIMEOUT", "B1_LEARN_TIMEOUT",
             "B1_REASONING_TIMEOUT")
    command = launch_command(root, inspected, job["run_id"], job["world_limit"],
                             environment["B1_QDRANT_URL"], environment["OLLAMA_URL"],
                             job_path=job_path, forwarded_env=names)
    plan = {"stage": "setup_only_not_benchmark_results", "image": inspected["Image"],
            "source_container": container, "command": command,
            "credential_delivery": "Docker --env OPENAI_API_KEY reads the launcher's environment; no value saved"}
    write_json(job_path.parent / "docker.launch.json", plan)
    return plan


def inside_benchmark(job_path: Path) -> int:
    from .doctor import doctor
    from .pln_runner import run_pln
    job = json.loads(job_path.read_text())
    root = Path.cwd()
    result = doctor(root, Path("/app"), Path(sys.executable), job_path.parent / "docker.doctor.json")
    if not result["ready"]:
        print("Docker dependency check failed; see docker.doctor.json", flush=True)
        return 2
    run_pln(root, Path("/app"), Path(sys.executable), Path(job["worlds_path"]),
            Path(job["expected_path"]), job["run_id"], job["world_limit"],
            dataset_kind=job["dataset_kind"], output_dir=Path(job["output_dir"]),
            request_budget_path=Path(job["request_budget_path"]))
    return 0  # Result accuracy/errors are reported in records and the shared summary.


def docker_pilot(root: Path, container: str, world_limit: int, qdrant_url: str,
                 ollama_url: str, prepare_only: bool = False) -> int:
    if not 1 <= world_limit <= 8:
        raise ValueError("world-limit must be between 1 and 8")
    env = configured_environment(root)
    settings = LLMSettings.from_env(env)
    if not env.get("OPENAI_API_KEY", "").strip():
        raise ValueError("Configure the OpenRouter key in this benchmark's .env first")
    inspection = subprocess.run(["docker", "inspect", "--type", "container", container],
                                capture_output=True, text=True, check=False)
    if inspection.returncode:
        raise RuntimeError("Cannot inspect the container. Run this command from a terminal with Docker access (sudo if required).")
    inspected = json.loads(inspection.stdout)[0]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-docker-" + uuid.uuid4().hex[:6]
    command = launch_command(root, inspected, run_id, world_limit, qdrant_url, ollama_url)
    plan = {"stage": "docker_setup_not_benchmark_results", "run_id": run_id,
            "source_container": container, "image": inspected["Image"],
            "world_limit": world_limit, "question_count": 2 * world_limit,
            "llm_settings": settings.public(), "command": command,
            "isolation": "New container; no live application volume; unique world collections; fresh query workers"}
    path = root / "artifacts/setup" / f"{run_id}.launch.json"
    write_json(path, plan)
    print(f"Launch plan: {path}", flush=True)
    if prepare_only:
        return 0
    print(f"Running {world_limit} worlds / {2 * world_limit} questions using the existing image.", flush=True)
    return subprocess.run(command, check=False).returncode


def inside(run_id: str, world_limit: int) -> int:
    from .doctor import doctor
    from .pln_runner import run_pilot
    root = Path.cwd()
    setup_path = root / "artifacts/setup" / f"{run_id}.doctor.json"
    result = doctor(root, Path("/app"), Path(sys.executable), setup_path)
    print(json.dumps({k: v for k, v in result.items() if k != "runtime_log"}, indent=2), flush=True)
    if not result["ready"]:
        print(f"Runtime check failed before any LLM call. Details: {setup_path}", flush=True)
        return 2
    output = run_pilot(root, Path("/app"), Path(sys.executable),
                       root / "data/pilot/worlds.json", root / "data/pilot/hand_expected.json", run_id, world_limit)
    print(f"Raw PLN-RAG results: {output / 'raw.md'}", flush=True)
    gate = json.loads((output / "gate.json").read_text())
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["status"] in ("TRIAL_COMPLETE", "PILOT_PASSED") else 2


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--inside-benchmark" and os.environ.get("B1_DOCKER_IMAGE"):
        raise SystemExit(inside_benchmark(Path(sys.argv[2])))
    if len(sys.argv) != 4 or sys.argv[1] != "--inside" or not os.environ.get("B1_DOCKER_IMAGE"):
        raise SystemExit("Use python3 -m b1 docker-pilot --container NAME from the benchmark folder")
    raise SystemExit(inside(sys.argv[2], int(sys.argv[3])))
