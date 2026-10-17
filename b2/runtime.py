"""B1-style repository mounts, one PLN container, and fresh Python workers."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

from b1.environment import redact, secret_values, file_sha256
from .dataset import write
from .worker import validate_job


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


def docker_command(identity, root, output, name):
    if any("," in str(p) for p in (root, output)):
        raise ValueError("Docker bind paths cannot contain commas")
    command = ["docker", "run", "--rm", "--pull", "never", "--init", "--name", name,
        "--network", "host", "--user", f"{os.geteuid()}:{os.getegid()}",
        "--mount", f"type=bind,src={root},dst={root}", "--workdir", str(root),
        "--env", "OPENAI_API_KEY", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "PYTHONUNBUFFERED=1", "--env", "B1_USE_ROOTLESS_RUNTIME=false",
        "--env", "LITELLM_LOCAL_MODEL_COST_MAP=True", "--entrypoint", "python3"]
    if not output.is_relative_to(root):
        command += ["--mount", f"type=bind,src={output},dst={output}"]
    for mount in identity["parser_mounts"]:
        command += ["--mount", f"type=bind,src={mount['source']},dst={mount['destination']},readonly"]
    # Import from the mounted working directory. Never forward host PYTHONPATH,
    # LD_LIBRARY_PATH or SWI_HOME_DIR: the image supplies its native dependencies.
    return command + [identity["image"], "-m", "b2.runtime", "--inside-pln", str(output / "pln_job.json")]


def clean_artifacts(operation, environment):
    secrets = secret_values(environment)
    for path in operation.rglob("*"):
        if path.is_file() and path.suffix in (".log", ".json", ".txt", ".body"):
            original = path.read_text(encoding="utf-8", errors="replace")
            clean = redact(original, secrets)
            if clean != original:
                path.write_text(clean, encoding="utf-8")


def finish_operation(operation, environment, started, failure=None, detail=None, exit_code=None):
    try:
        row = json.loads((operation / "output.json").read_text())
    except (OSError, ValueError):
        row = {}
        failure = failure or "missing_worker_output"
    if not failure and exit_code and row.get("status") == "OK":
        failure = "worker_exit_error"
    if failure:
        row.update(status="ERROR", attempted=True, error_kind=failure, used_ids=None)
        if detail:
            row["error_detail"] = detail
    row["timed_out"] = row.get("timed_out", False) or row.get("error_kind") == "timeout"
    row["worker_seconds"] = time.perf_counter() - started
    if (operation / "requests.json").exists():
        row["llm_http_calls"] = json.loads((operation / "requests.json").read_text())
    row = json.loads(redact(json.dumps(row, ensure_ascii=False), secret_values(environment)))
    write(operation / "output.json", row)
    clean_artifacts(operation, environment)
    return row


class OperationTimeout(Exception):
    """Distinct from network exceptions caught by the baseline HTTP client."""


@contextmanager
def deadline(seconds):
    """Bound a baseline operation in the main process, including pacing/indexing."""
    def expired(signum, frame):
        raise OperationTimeout()
    previous = signal.signal(signal.SIGALRM, expired)
    timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if timer[0]:
            signal.setitimer(signal.ITIMER_REAL, max(.000001, timer[0] - (time.monotonic() - started)), timer[1])


def run_baseline(job, operation, environment, timeout):
    """Call the unchanged explanation adapter directly, retaining operation audits."""
    from .baseline import run
    validate_job(job)
    operation.mkdir(parents=True, exist_ok=True)
    write(operation / "job.json", job)
    write(operation / "launch.json", {"execution": "main_process", "timeout_seconds": timeout})
    started, failure, detail = time.perf_counter(), None, None
    try:
        with (operation / "raw.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            try:
                with deadline(timeout):
                    run(job, environment, operation)
            except OperationTimeout:
                failure = "timeout"
            except Exception as exc:
                traceback.print_exc()
                failure, detail = type(exc).__name__, str(exc)
    except KeyboardInterrupt:
        finish_operation(operation, environment, started, "interrupted")
        raise
    return finish_operation(operation, environment, started, failure, detail)


class Runtime:
    def __init__(self, root, output, environment, *, pln_python, pln_runtime, container):
        self.root, self.output, self.env = root, output, environment
        self.pln_python, self.pln_runtime, self.container = pln_python, pln_runtime, container
        self.identity = None

    def run_pln(self, job):
        from .pln_runner import run_pln
        write(self.output / "pln_job.json", job)
        if self.pln_runtime == "local":
            return run_pln(job, self.env, self.execute)
        self.identity = inspect_container(self.container)
        write(self.output / "runtime.json", self.identity)
        name = "b2-" + job["run_id"] + "-" + uuid.uuid4().hex[:8]
        command = docker_command(self.identity, self.root, self.output, name)
        write(self.output / "docker.launch.json", {"command": command, **self.identity,
            "credential_delivery": "Docker --env OPENAI_API_KEY reads the launcher's environment; no value saved"})
        process = None
        try:
            with (self.output / "docker.log").open("w") as log:
                process = subprocess.Popen(command, cwd=self.root, env=self.env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if process.wait():
                    raise RuntimeError("PLN container exited unsuccessfully; see docker.log")
        finally:
            # --rm handles normal exit; force removal also covers interruption or
            # a Docker client failure while the container is still running.
            try:
                if process is not None:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30, check=False)
            finally:
                clean_artifacts(self.output, self.env)

    def execute(self, job, operation, timeout):
        """One learning/query Python process; called locally or inside the container."""
        validate_job(job)
        if job["system"] != "pln_rag":
            raise ValueError("Baselines run in the main benchmark process")
        operation.mkdir(parents=True, exist_ok=False)
        write(operation / "job.json", job)
        command = [str(self.pln_python), "-m", "b2.worker", "--job", str(operation / "job.json")]
        write(operation / "launch.json", {"command": command, "timeout_seconds": timeout})
        environment = dict(self.env)
        # Keep the image's full native search path. cwd makes the repository
        # importable without replacing PYTHONPATH, including in child processes.
        environment["DSPY_CACHEDIR"] = str(operation / "cache")
        started, failure, detail, process = time.perf_counter(), None, None, None
        try:
            with (operation / "raw.log").open("w") as log:
                process = subprocess.Popen(command, cwd=self.root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    failure = "timeout"
        except KeyboardInterrupt:
            failure = "interrupted"
            raise
        except Exception as exc:
            failure, detail = type(exc).__name__, str(exc)
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            row = finish_operation(operation, environment, started, failure, detail,
                                   process.returncode if process is not None else None)
        return row


def inside_pln(job_path):
    job = json.loads(job_path.read_text())
    runtime = Runtime(Path.cwd(), job_path.parent, dict(os.environ),
                      pln_python=Path(sys.executable), pln_runtime="local", container=None)
    runtime.run_pln(job)
    return 0  # Individual failures remain in the durable records, as in B1.


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--inside-pln":
        raise SystemExit("Use python3 -m b2 benchmark")
    raise SystemExit(inside_pln(Path(sys.argv[2])))
