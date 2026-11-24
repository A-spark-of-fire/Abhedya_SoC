"""
ExecuTorch Llama 3.2 Engine Wrapper for Abhedya SoC
Bharat AI-SoC Challenge — Problem Statement 3: On-Device Security

Runs the pre-quantized Meta Llama 3.2 1B-Instruct model (4-bit XNNPACK PTE)
via the native ExecuTorch C++ CLI runner on Raspberry Pi 5 / Arm64.
Includes robust timeout handling and a deterministic rule-based fallback mechanism.

Key optimizations & safety features (Problems 1, 2, 3):
- Robust multi-byte streaming UTF-8 decoding with errors="replace" and 180s timeout.
- Filters out stdout execution statistics lines emitted by llama_main.
- max_new_tokens reduced to 120 (trimmed from 256) to lower generation latency from ~41s to ~12-15s.
- ExecuTorch Llama generation is invoked ONLY when status == "ALERT"; benign/idle events bypass LLM.
- Generates 1-sentence threat summary and stitches with deterministic safe remediation templates.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

from llama_prompt_builder import (
    assemble_advisory,
    build_llama_prompt,
    clean_llm_response,
    filter_executorch_stats,
    generate_grounded_fallback_advisory,
)

logger = logging.getLogger("executorch_runner")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models" / "llama"
DEFAULT_USER_MODELS_DIR = Path.home() / "models_llm"


def find_file_in_candidates(candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p.resolve()
    return None


class ExecuTorchLlamaEngine:
    """
    On-device reasoning engine wrapping ExecuTorch Llama 3.2 (1B-Instruct 4-bit PTE).
    Compiles Tier 1 telemetry into actionable plain-English security advisories.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        tokenizer_path: Optional[str | Path] = None,
        runner_path: Optional[str | Path] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 120,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.temperature = float(temperature)
        # Reduced max_new_tokens to 120 (Problem 3) for ~12-15s generation on Arm CPU
        self.max_new_tokens = int(max_new_tokens)
        self.timeout_seconds = float(timeout_seconds)

        # 1. Resolve ExecuTorch PTE model file
        pte_candidates = [
            Path(model_path) if model_path else None,
            Path(os.environ.get("EXECUTORCH_PTE_PATH", "")),
            Path(os.environ.get("NIDS_EXECUTORCH_PTE_PATH", "")),
            DEFAULT_MODELS_DIR / "xnnpack" / "Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte",
            DEFAULT_MODELS_DIR / "Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte",
            DEFAULT_USER_MODELS_DIR / "xnnpack" / "Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte",
            DEFAULT_USER_MODELS_DIR / "Llama-3.2-1B-Instruct-8da4w-gptq-4k.pte",
        ]
        self.resolved_model_path = find_file_in_candidates([p for p in pte_candidates if p])

        # 2. Resolve tokenizer.json
        tok_candidates = [
            Path(tokenizer_path) if tokenizer_path else None,
            Path(os.environ.get("EXECUTORCH_TOKENIZER_PATH", "")),
            Path(os.environ.get("NIDS_EXECUTORCH_TOKENIZER_PATH", "")),
            DEFAULT_MODELS_DIR / "tokenizer.json",
            DEFAULT_USER_MODELS_DIR / "tokenizer.json",
        ]
        self.resolved_tokenizer_path = find_file_in_candidates([p for p in tok_candidates if p])

        # 3. Resolve ExecuTorch C++ CLI runner binary (llama_main / executorch_llama_runner)
        self.resolved_runner_path = self._find_runner_binary(runner_path)

        # Log active mode
        if self.is_native_available():
            logger.info(
                f"[ExecuTorch] Native Arm64 runner found at: {self.resolved_runner_path} "
                f"(Model: {self.resolved_model_path.name})"
            )
        else:
            reason = []
            if not self.resolved_runner_path:
                reason.append("C++ runner binary ('llama_main') not found")
            if not self.resolved_model_path:
                reason.append("PTE model file not downloaded")
            if not self.resolved_tokenizer_path:
                reason.append("tokenizer.json not found")
            logger.info(
                f"[ExecuTorch] Running with Deterministic Edge Copilot Fallback ({', '.join(reason)})."
            )

    def _find_runner_binary(self, override: Optional[str | Path]) -> Optional[Path]:
        if override:
            p = Path(override)
            if p.exists() and p.is_file():
                return p.resolve()

        env_runner = os.environ.get("EXECUTORCH_RUNNER_PATH") or os.environ.get("NIDS_EXECUTORCH_RUNNER_PATH")
        if env_runner:
            p = Path(env_runner)
            if p.exists() and p.is_file():
                return p.resolve()

        for binary_name in ["llama_main", "executorch_llama_runner"]:
            which_res = shutil.which(binary_name)
            if which_res:
                return Path(which_res).resolve()

        known_paths = [
            Path("/opt/nids/bin/llama_main"),
            Path("/usr/local/bin/llama_main"),
            Path("/usr/bin/llama_main"),
            DEFAULT_MODELS_DIR / "llama_main",
            DEFAULT_MODELS_DIR / "executorch_llama_runner",
            PROJECT_ROOT / "llama_main",
            PROJECT_ROOT / "executorch_llama_runner",
        ]
        return find_file_in_candidates(known_paths)

    def is_native_available(self) -> bool:
        """Returns True if the C++ runner binary, the .pte model, and tokenizer.json are all present."""
        return (
            self.resolved_runner_path is not None
            and self.resolved_model_path is not None
            and self.resolved_tokenizer_path is not None
        )

    def get_status_info(self) -> Dict[str, Any]:
        native = self.is_native_available()
        return {
            "native_engine_available": native,
            "backend": "ExecuTorch Arm64 XNNPACK (Native)" if native else "Deterministic Edge Copilot (Grounded Fallback)",
            "runner_path": str(self.resolved_runner_path) if self.resolved_runner_path else "Not Installed",
            "model_path": str(self.resolved_model_path) if self.resolved_model_path else "Missing",
            "tokenizer_path": str(self.resolved_tokenizer_path) if self.resolved_tokenizer_path else "Missing",
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "timeout_seconds": self.timeout_seconds,
        }

    def generate_advisory(self, telemetry_json: Dict[str, Any]) -> str:
        """
        Generates the 3-part plain-English advisory from telemetry JSON.
        Invokes ExecuTorch Llama generation ONLY when status == "ALERT".
        Benign and idle events immediately bypass the LLM (0 LLM compute overhead).
        """
        t0 = time.perf_counter()
        status = telemetry_json.get("status", "CLEAN")
        active_threats = telemetry_json.get("active_threats_count", 0)

        # Benign and idle events must immediately bypass the LLM (Problem 3)
        if status != "ALERT" or active_threats == 0:
            return generate_grounded_fallback_advisory(telemetry_json)

        prompt = build_llama_prompt(telemetry_json)

        # 1. Native ExecuTorch execution attempt
        if self.is_native_available():
            try:
                advisory = self._run_native_executorch(prompt, telemetry_json=telemetry_json)
                if advisory and len(advisory.strip()) > 20:
                    latency = (time.perf_counter() - t0) * 1000.0
                    logger.info(f"[ExecuTorch] Generated native advisory in {latency:.2f} ms")
                    return advisory
            except Exception as exc:
                logger.warning(
                    f"[ExecuTorch] Native execution failed ({exc}). Falling back to grounded template advisory."
                )

        # 2. Grounded Fallback Mechanism
        advisory = generate_grounded_fallback_advisory(telemetry_json)
        latency = (time.perf_counter() - t0) * 1000.0
        logger.debug(f"[ExecuTorch] Grounded fallback advisory generated in {latency:.2f} ms")
        return advisory

    def _run_native_executorch(self, prompt: str, telemetry_json: Optional[Dict[str, Any]] = None) -> str:
        """Executes the compiled llama_main CLI runner via temporary prompt file."""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_file = Path(f.name)

        try:
            cmd = [
                str(self.resolved_runner_path),
                "--model_path",
                str(self.resolved_model_path),
                "--tokenizer_path",
                str(self.resolved_tokenizer_path),
                "--prompt_file",
                str(prompt_file),
                "--temperature",
                str(self.temperature),
                "--max_new_tokens",
                str(self.max_new_tokens),
            ]

            logger.info(f"Invoking ExecuTorch runner: {' '.join(cmd[:3])} ...")
            # Problem 1: Handle streaming multi-byte UTF-8 sequences and emojis gracefully
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
                check=False,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"ExecuTorch runner exited with code {result.returncode}: {result.stderr.strip()[:300]}"
                )

            # Strip prompt echo if present and extract generated response
            output = result.stdout.strip()
            if "<|start_header_id|>assistant<|end_header_id|>" in output:
                output = output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

            # Filter out execution statistics lines emitted by llama_main (e.g. [INFO], tokens/sec, prefill time)
            cleaned = clean_llm_response(output)
            filtered = filter_executorch_stats(cleaned)

            if telemetry_json:
                return assemble_advisory(filtered, telemetry_json)
            return filtered

        finally:
            if prompt_file.exists():
                try:
                    prompt_file.unlink()
                except Exception:
                    pass
