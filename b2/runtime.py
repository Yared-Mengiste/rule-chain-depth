"""Narrow worker mounts and fresh processes with hard deadlines."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from b1.environment import redact, secret_values, file_sha256
from .dataset import write
from .worker import validate_job

CODE_FILES = {
    "b1": ("__init__.py", "environment.py", "io.py", "model.py", "llm_control.py", "snapshot.py", "baseline.py"),
    "b2": ("__init__.py", "dataset.py", "worker.py", "pln_worker.py", "baseline.py", "evidence.py"),
}


def stage_code(destination):
    root = Path(__file__).resolve().parents[1]
    for package, names in CODE_FILES.items():
        for name in names:
            target = destination / package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / package / name, target)
            target.chmod(0o444)


def inspect_container(container):
    result = subprocess.run(["docker", "inspect", "--type", "container", container],
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError("Cannot inspect PLN container; check Docker access and --container")
    info = json.loads(result.stdout)[0]
    if not info.get("State", {}).get("Running") or not info.get("Image", "").startswith("sha256:"):
        raise ValueError("Expected a running container with an immutable image ID")
    mounts = []
    for mount in info.get("Mounts", []):
        if mount.get("Destination") not in ("/app/data/simba_canonical_pln.json", "/app/data/simba_all.json"):
            continue
        source = Path(mount["Source"])
        if mount["Type"] != "bind" or not source.is_file() or "," in str(source):
            raise ValueError("Cannot mount inspected parser artifact")
        mounts.append({"source": str(source), "destination": mount["Destination"], "sha256": file_sha256(source)})
    return {"image": info["Image"], "source_container": container, "parser_mounts": mounts}


def docker_command(identity, code, exchange, operation, name):
    if any("," in str(p) for p in (code, exchange, operation)):
        raise ValueError("Docker bind paths cannot contain commas")
    command = ["docker", "run", "--rm", "--pull", "never", "--init", "--name", name,
        "--network", "host", "--user", f"{os.geteuid()}:{os.getegid()}",
        "--mount", f"type=bind,src={code},dst={code},readonly",
        "--mount", f"type=bind,src={exchange},dst={exchange}",
        "--workdir", str(operation), "--env", "OPENAI_API_KEY", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONUNBUFFERED=1", "--env", "PYTHONPATH=" + str(code),
        "--env", "DSPY_CACHEDIR=" + str(operation / "cache"),
        "--env", "LITELLM_LOCAL_MODEL_COST_MAP=True", "--entrypoint", "python3"]
    for mount in identity["parser_mounts"]:
        command += ["--mount", f"type=bind,src={mount['source']},dst={mount['destination']},readonly"]
    return command + [identity["image"], "-m", "b2.worker", "--job", str(operation / "job.json")]


class Runtime:
    def __init__(self, output, environment, *, python, pln_python, pln_runtime, container):
        self.output, self.env = output, environment
        self.code, self.exchange = output / "worker-code", output / "workers"
        self.exchange.mkdir()
        stage_code(self.code)
        self.python, self.pln_python = python, pln_python
        self.pln_runtime, self.container = pln_runtime, container
        self.identity = None

    def execute(self, job, operation, timeout):
        validate_job(job)
        operation.mkdir(parents=True, exist_ok=False)
        write(operation / "job.json", job)
        native_docker = job["system"] == "pln_rag" and self.pln_runtime == "docker"
        name = "b2-" + uuid.uuid4().hex
        if native_docker:
            if self.identity is None:
                self.identity = inspect_container(self.container)
                write(self.output / "runtime.json", self.identity)
            command = docker_command(self.identity, self.code, self.exchange, operation, name)
        else:
            command = [str(self.pln_python if job["system"] == "pln_rag" else self.python),
                       "-m", "b2.worker", "--job", str(operation / "job.json")]
        write(operation / "launch.json", {"command": command, "timeout_seconds": timeout})
        environment = dict(self.env)
        # No checkout/dataset path in the import path or current directory.
        runtime_paths = [p for p in environment.get("PYTHONPATH", "").split(os.pathsep)
                         if p and "/runtime/deps/" in p]
        environment["PYTHONPATH"] = os.pathsep.join([str(self.code), *runtime_paths])
        environment["DSPY_CACHEDIR"] = str(operation / "cache")
        started = time.perf_counter()
        failure, process = None, None
        try:
            with (operation / "raw.log").open("w") as log:
                process = subprocess.Popen(command, cwd=operation, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    failure = "timeout"
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if native_docker:
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
            secrets = secret_values(environment)
            for path in operation.rglob("*"):
                if path.is_file() and path.suffix in (".log", ".json", ".txt", ".body"):
                    original = path.read_text(encoding="utf-8", errors="replace")
                    clean = redact(original, secrets)
                    if clean != original:
                        path.write_text(clean, encoding="utf-8")
        row = {}
        try:
            row = json.loads((operation / "output.json").read_text())
        except (OSError, ValueError):
            failure = failure or "missing_worker_output"
        if failure or (process.returncode and row.get("status") == "OK"):
            row.update(status="ERROR", attempted=True, error_kind=failure or "worker_exit_error")
            # Never score a partial proof as a completed answer after a deadline.
            row["used_ids"] = None
        row["timed_out"] = row.get("timed_out", False) or row.get("error_kind") == "timeout"
        row["worker_seconds"] = time.perf_counter() - started
        if (operation / "requests.json").exists():
            row["llm_http_calls"] = json.loads((operation / "requests.json").read_text())
        return row
