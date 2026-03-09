#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import random
import re
import statistics
import string
import subprocess
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

TAG_RE = re.compile(r"<\|[^|>]+\|>")
PUNCT_TABLE = str.maketrans("", "", string.punctuation + "，。！？；：、‘’“”（）《》【】—…·、\t\n\r ")
RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SWAP_RE = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")

SHORT_TEXTS = [
    ("zh_short", "今天我们测试语音服务。"),
]

CN_SEGMENTS = [
    "今天我们测试语音服务。",
    "系统会记录资源占用。",
    "如果接口稳定，我们继续测试并发。",
    "我们希望语音清晰自然。",
]
EN_SEGMENTS = [
    "speech service is ready.",
]


def generate_long_mixed_text(target_length: int = 800) -> str:
    phrase = "今天我们测试语音服务。speech service is ready。"
    text = phrase * ((target_length // len(phrase)) + 2)
    return text[:target_length]


def sanitize_asr_text(text: str) -> str:
    cleaned = TAG_RE.sub("", text or "")
    cleaned = cleaned.replace("<", "").replace(">", "")
    return cleaned.strip()


def normalize_text(text: str) -> str:
    cleaned = sanitize_asr_text(text).lower()
    cleaned = cleaned.translate(PUNCT_TABLE)
    return cleaned


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


@dataclass
class RequestResult:
    label: str
    text: str
    response_format: str
    ok: bool
    status_code: int
    elapsed_s: float
    bytes_len: int
    asr_text: str
    sanitized_asr_text: str
    similarity: float
    passed: bool
    error: str | None
    output_file: str | None


class ResourceSampler:
    def __init__(self, ssh_target: str, container_name: str, out_dir: Path) -> None:
        self.ssh_target = ssh_target
        self.container_name = container_name
        self.out_dir = out_dir
        self.processes: list[subprocess.Popen[str]] = []
        self.handles = []

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tegra_handle = (self.out_dir / "tegrastats.log").open("w")
        docker_handle = (self.out_dir / "docker-stats.log").open("w")
        self.handles.extend([tegra_handle, docker_handle])
        self.processes.append(
            subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    self.ssh_target,
                    "bash -lc 'tegrastats --interval 1000'",
                ],
                stdout=tegra_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
        self.processes.append(
            subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "StrictHostKeyChecking=no",
                    self.ssh_target,
                    (
                        "bash -lc 'while true; do "
                        f"docker stats --no-stream --format \"{{{{json .}}}}\" {self.container_name}; "
                        "sleep 1; done'"
                    ),
                ],
                stdout=docker_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.processes.clear()
        for handle in self.handles:
            handle.close()
        self.handles.clear()


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def ensure_health(url: str, timeout: int = 5) -> None:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()


def synthesize(tts_url: str, text: str, response_format: str, out_dir: Path) -> tuple[bytes, float, int, Path]:
    output_file = out_dir / f"{uuid.uuid4().hex}.{response_format}"
    started = time.perf_counter()
    response = requests.post(
        tts_url,
        json={
            "model": "openaudio-s1-mini",
            "input": text,
            "response_format": response_format,
        },
        timeout=600,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    output_file.write_bytes(response.content)
    return response.content, elapsed, response.status_code, output_file


def transcribe_file(asr_url: str, audio_path: Path, model: str = "sensevoice-small") -> str:
    with audio_path.open("rb") as handle:
        response = requests.post(
            asr_url,
            files={"file": (audio_path.name, handle, "application/octet-stream")},
            data={"model": model, "response_format": "json"},
            timeout=600,
        )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return str(payload.get("text", payload.get("result", "")))
    return str(payload)


def split_wav(audio_path: Path, chunk_seconds: int = 15) -> list[Path]:
    chunk_paths: list[Path] = []
    with wave.open(str(audio_path), "rb") as src:
        framerate = src.getframerate()
        channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        comptype = src.getcomptype()
        compname = src.getcompname()
        frames_per_chunk = framerate * chunk_seconds
        total_frames = src.getnframes()
        index = 0
        while src.tell() < total_frames:
            frames = src.readframes(frames_per_chunk)
            if not frames:
                break
            chunk_path = audio_path.with_name(f"{audio_path.stem}-chunk-{index:03d}{audio_path.suffix}")
            with wave.open(str(chunk_path), "wb") as dst:
                dst.setnchannels(channels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(framerate)
                dst.setcomptype(comptype, compname)
                dst.writeframes(frames)
            chunk_paths.append(chunk_path)
            index += 1
    return chunk_paths


def transcribe(asr_url: str, audio_path: Path, model: str = "sensevoice-small") -> str:
    if audio_path.suffix.lower() != ".wav":
        return transcribe_file(asr_url, audio_path, model)

    with wave.open(str(audio_path), "rb") as src:
        duration = src.getnframes() / float(src.getframerate())
    if duration <= 15:
        return transcribe_file(asr_url, audio_path, model)

    texts: list[str] = []
    chunk_paths = split_wav(audio_path, chunk_seconds=15)
    try:
        for chunk_path in chunk_paths:
            texts.append(sanitize_asr_text(transcribe_file(asr_url, chunk_path, model)))
    finally:
        for chunk_path in chunk_paths:
            chunk_path.unlink(missing_ok=True)
    return "".join(texts)


def run_one(tts_url: str, asr_url: str, label: str, text: str, response_format: str, out_dir: Path, threshold: float) -> RequestResult:
    try:
        _, elapsed, status_code, output_file = synthesize(tts_url, text, response_format, out_dir)
        asr_text = transcribe(asr_url, output_file)
        sim = similarity(text, asr_text)
        return RequestResult(
            label=label,
            text=text,
            response_format=response_format,
            ok=True,
            status_code=status_code,
            elapsed_s=elapsed,
            bytes_len=output_file.stat().st_size,
            asr_text=asr_text,
            sanitized_asr_text=sanitize_asr_text(asr_text),
            similarity=sim,
            passed=sim >= threshold,
            error=None,
            output_file=str(output_file),
        )
    except Exception as exc:  # noqa: BLE001
        return RequestResult(
            label=label,
            text=text,
            response_format=response_format,
            ok=False,
            status_code=0,
            elapsed_s=0.0,
            bytes_len=0,
            asr_text="",
            sanitized_asr_text="",
            similarity=0.0,
            passed=False,
            error=str(exc),
            output_file=None,
        )


def write_request_results(results: list[RequestResult], out_csv: Path, out_json: Path) -> None:
    rows = [asdict(item) for item in results]
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["label"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def summarize(results: list[RequestResult]) -> dict[str, Any]:
    latencies = [item.elapsed_s for item in results if item.ok]
    passed = [item for item in results if item.passed]
    return {
        "requests": len(results),
        "ok": sum(1 for item in results if item.ok),
        "passed": len(passed),
        "success_rate": round(sum(1 for item in results if item.ok) / len(results), 4) if results else 0.0,
        "pass_rate": round(len(passed) / len(results), 4) if results else 0.0,
        "avg_latency_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.5), 4) if latencies else 0.0,
        "p95_latency_s": round(percentile(latencies, 0.95), 4) if latencies else 0.0,
        "throughput_rps": round(len(latencies) / sum(latencies), 4) if latencies and sum(latencies) > 0 else 0.0,
    }


def parse_size_to_mb(value: str) -> float:
    value = value.strip()
    match = re.match(r"([0-9.]+)([KMG]i?B)", value)
    if not match:
        return 0.0
    number = float(match.group(1))
    unit = match.group(2)
    factors = {
        "KiB": 1 / 1024,
        "MiB": 1,
        "GiB": 1024,
        "KB": 1 / 1000,
        "MB": 1,
        "GB": 1000,
    }
    return number * factors.get(unit, 0.0)


def summarize_docker_stats(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    cpu_values: list[float] = []
    mem_values: list[float] = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        cpu = str(payload.get("CPUPerc", "0")).rstrip("%")
        mem_usage = str(payload.get("MemUsage", "0MiB / 0GiB")).split("/")[0].strip()
        try:
            cpu_values.append(float(cpu))
        except ValueError:
            pass
        mem_values.append(parse_size_to_mb(mem_usage))
    return {
        "docker_cpu_avg_pct": round(statistics.mean(cpu_values), 2) if cpu_values else 0.0,
        "docker_cpu_peak_pct": round(max(cpu_values), 2) if cpu_values else 0.0,
        "docker_mem_peak_mb": round(max(mem_values), 2) if mem_values else 0.0,
    }


def summarize_tegrastats(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    ram_used: list[int] = []
    swap_used: list[int] = []
    gr3d_values: list[int] = []
    for line in log_path.read_text().splitlines():
        if match := RAM_RE.search(line):
            ram_used.append(int(match.group(1)))
        if match := SWAP_RE.search(line):
            swap_used.append(int(match.group(1)))
        if match := GR3D_RE.search(line):
            gr3d_values.append(int(match.group(1)))
    return {
        "jetson_ram_peak_mb": round(max(ram_used), 2) if ram_used else 0.0,
        "jetson_swap_peak_mb": round(max(swap_used), 2) if swap_used else 0.0,
        "jetson_gr3d_peak_pct": round(max(gr3d_values), 2) if gr3d_values else 0.0,
    }


def summarize_stage_resources(resources_dir: Path) -> dict[str, float]:
    summary = {}
    summary.update(summarize_docker_stats(resources_dir / "docker-stats.log"))
    summary.update(summarize_tegrastats(resources_dir / "tegrastats.log"))
    if summary:
        (resources_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def collect_failure_artifacts(ssh_target: str, container_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "docker-logs.txt": f"docker logs --tail 500 {container_name}",
        "docker-inspect.json": f"docker inspect {container_name}",
        "tegrastats-snapshot.txt": "timeout 10s tegrastats --interval 1000",
        "docker-stats-snapshot.txt": f"docker stats --no-stream {container_name}",
    }
    for filename, command in commands.items():
        target = out_dir / filename
        with target.open("w") as handle:
            subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=no", ssh_target, f"bash -lc {shlex_quote(command)}"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )


def build_test_matrix(response_formats: list[str]) -> list[tuple[str, str, str]]:
    matrix: list[tuple[str, str, str]] = []
    for fmt in response_formats:
        for label, text in SHORT_TEXTS:
            matrix.append((f"{label}_{fmt}", text, fmt))
    return matrix


def run_concurrency_stage(
    tts_url: str,
    asr_url: str,
    concurrency: int,
    jobs: list[tuple[str, str, str]],
    stage_dir: Path,
    threshold: float,
    ssh_target: str,
    container_name: str,
    sample_resources: bool,
) -> tuple[list[RequestResult], dict[str, Any]]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    sampler = None
    if sample_resources:
        sampler = ResourceSampler(ssh_target, container_name, stage_dir / "resources")
        sampler.start()

    try:
        started = time.perf_counter()
        results: list[RequestResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(run_one, tts_url, asr_url, label, text, fmt, stage_dir, threshold)
                for label, text, fmt in jobs
            ]
            for future in as_completed(futures):
                results.append(future.result())
        wall = time.perf_counter() - started
    finally:
        if sampler is not None:
            sampler.stop()

    summary = summarize(results)
    summary["concurrency"] = concurrency
    summary["wall_time_s"] = round(wall, 4)
    summary["chars_total"] = sum(len(item.text) for item in results)
    summary["chars_per_second"] = round(summary["chars_total"] / wall, 4) if wall > 0 else 0.0
    if sample_resources:
        summary.update(summarize_stage_resources(stage_dir / "resources"))
    write_request_results(results, stage_dir / "requests.csv", stage_dir / "requests.json")
    (stage_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return results, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and validate OpenAI-compatible TTS on Jetson")
    parser.add_argument("--tts-url", default="http://192.168.1.230:8080/v1/audio/speech")
    parser.add_argument("--tts-health-url", default="http://192.168.1.230:8080/v1/health")
    parser.add_argument("--asr-url", default="http://192.168.1.230:10001/v1/audio/transcriptions")
    parser.add_argument("--asr-health-url", default="http://192.168.1.230:10001/stream/v1/asr/health")
    parser.add_argument("--ssh-target", default="nvidia@192.168.1.230")
    parser.add_argument("--container-name", default="fish-speech-jetson-server-1")
    parser.add_argument("--output-dir", default="jetson-bench-results")
    parser.add_argument("--similarity-threshold", type=float, default=0.98)
    parser.add_argument("--concurrency", default="1,2,4,6,8,16,32")
    parser.add_argument("--response-formats", default="wav")
    parser.add_argument("--skip-resource-sampling", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) / time.strftime("run-%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_health(args.tts_health_url)
    ensure_health(args.asr_health_url)

    long_text = generate_long_mixed_text(800)
    (output_dir / "long_text.txt").write_text(long_text)
    formats = [item.strip() for item in args.response_formats.split(",") if item.strip()]
    matrix = build_test_matrix(formats)

    warmup_dir = output_dir / "warmup"
    warmup_jobs = matrix[: max(2, len(formats))]
    warmup_results, warmup_summary = run_concurrency_stage(
        args.tts_url,
        args.asr_url,
        1,
        warmup_jobs,
        warmup_dir,
        args.similarity_threshold,
        args.ssh_target,
        args.container_name,
        not args.skip_resource_sampling,
    )
    all_summaries: list[dict[str, Any]] = [{"stage": "warmup", **warmup_summary}]
    failures: list[RequestResult] = [item for item in warmup_results if not item.passed]

    concurrency_levels = [int(item.strip()) for item in args.concurrency.split(",") if item.strip()]
    for concurrency in concurrency_levels:
        stage_dir = output_dir / f"concurrency-{concurrency}"
        jobs = []
        for index in range(concurrency):
            label, text, fmt = matrix[index % len(matrix)]
            jobs.append((f"{label}_c{concurrency}_{index}", text, fmt))
        results, summary = run_concurrency_stage(
            args.tts_url,
            args.asr_url,
            concurrency,
            jobs,
            stage_dir,
            args.similarity_threshold,
            args.ssh_target,
            args.container_name,
            not args.skip_resource_sampling,
        )
        all_summaries.append({"stage": f"concurrency-{concurrency}", **summary})
        failures.extend([item for item in results if not item.passed])

    for fmt in formats:
        long_stage_dir = output_dir / f"long-text-{fmt}"
        long_jobs = [(f"mix_long_{fmt}", long_text, fmt)]
        long_results, long_summary = run_concurrency_stage(
            args.tts_url,
            args.asr_url,
            1,
            long_jobs,
            long_stage_dir,
            args.similarity_threshold,
            args.ssh_target,
            args.container_name,
            not args.skip_resource_sampling,
        )
        long_summary["chars_total"] = len(long_text)
        long_summary["validation_note"] = "long_text_similarity_recorded_but_not_gated"
        all_summaries.append({"stage": f"long-text-{fmt}", **long_summary})
        failures.extend([item for item in long_results if not item.ok])

    (output_dir / "summaries.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2))
    fieldnames: list[str] = []
    for summary in all_summaries:
        for key in summary.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with (output_dir / "summaries.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summaries)

    if failures:
        failure_dir = output_dir / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "failed_requests.json").write_text(
            json.dumps([asdict(item) for item in failures], ensure_ascii=False, indent=2)
        )
        collect_failure_artifacts(args.ssh_target, args.container_name, failure_dir)
        print(f"Benchmark finished with failures. See {failure_dir}")
        return 1

    print(f"Benchmark succeeded. Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
