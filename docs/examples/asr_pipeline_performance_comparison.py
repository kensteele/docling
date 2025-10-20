#!/usr/bin/env python3
"""
Performance comparison between CPU and MLX Whisper on Apple Silicon.

This script compares the performance of:
1. Native Whisper (forced to CPU)
2. MLX Whisper (Apple Silicon optimized)

Both use the same model size for fair comparison.
"""

import argparse
import csv
import gc
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the repository root to the path so we can import docling
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.datamodel.pipeline_options_asr_model import (
    InferenceAsrFramework,
    InlineAsrMlxWhisperOptions,
    InlineAsrNativeWhisperOptions,
)
from docling.document_converter import AudioFormatOption, DocumentConverter
from docling.pipeline.asr_pipeline import AsrPipeline


def cleanup_mlx_resources():
    """Clean up MLX runtime resources without deleting model files."""
    try:
        # Try to import MLX core for cleanup
        import mlx.core as mx

        # Clear MLX computation graph and free GPU memory
        mx.eval([])  # This clears the computation graph

        # Additional cleanup - clear any cached arrays
        try:
            mx.clear_cache()
        except Exception:
            pass  # Ignore if clear_cache is not available

    except ImportError:
        pass  # MLX not available, skip cleanup
    except Exception:
        pass  # Ignore cleanup errors

    # Force Python garbage collection to free object references
    gc.collect()

    # Comprehensive multiprocessing cleanup to prevent semaphore leaks
    try:
        import multiprocessing
        import os
        import signal

        # Get all active child processes and terminate them properly
        active_children = multiprocessing.active_children()
        for child in active_children:
            try:
                # Try to terminate gracefully first
                child.terminate()
                child.join(timeout=1.0)  # Wait up to 1 second

                # If still alive, force kill
                if child.is_alive():
                    os.kill(child.pid, signal.SIGKILL)
                    child.join(timeout=0.5)
            except Exception:
                pass  # Ignore cleanup errors for individual processes

        # Clear any remaining multiprocessing state
        try:
            # Force cleanup of multiprocessing module state
            if hasattr(multiprocessing, "_cleanup"):
                multiprocessing._cleanup()
        except Exception:
            pass

    except Exception:
        pass

    # Additional cleanup for any remaining resources
    try:
        # Clear any remaining semaphores or shared memory
        import threading

        # Force cleanup of threading resources
        for thread in threading.enumerate():
            if thread != threading.current_thread() and not thread.daemon:
                try:
                    thread.join(timeout=0.1)
                except Exception:
                    pass
    except Exception:
        pass

    # Longer delay to allow resources to be properly released
    time.sleep(0.5)


def aggressive_cleanup_mlx_resources():
    """More aggressive cleanup for comprehensive tests."""
    # Multiple cleanup passes
    for i in range(3):
        cleanup_mlx_resources()
        if i < 2:  # Don't sleep after the last cleanup
            time.sleep(1.0)  # Longer delay between passes


# Complete MLX model registry
MLX_MODEL_REGISTRY = {
    "tiny": [
        "mlx-community/whisper-tiny-mlx",
        "mlx-community/whisper-tiny-mlx-q4",
        # "mlx-community/whisper-tiny-mlx-fp32",
        "mlx-community/whisper-tiny-mlx-8bit",
        "mlx-community/whisper-tiny-mlx-4bit",
        "mlx-community/whisper-tiny.en-mlx",
        "mlx-community/whisper-tiny.en-mlx-q4",
        "mlx-community/whisper-tiny.en-mlx-4bit",
        "mlx-community/whisper-tiny.en-mlx-8bit",
        # "mlx-community/whisper-tiny.en-mlx-fp32",
    ],
    "small": [
        "mlx-community/whisper-small-mlx",
        "mlx-community/whisper-small-mlx-q4",
        # "mlx-community/whisper-small-mlx-fp32",
        "mlx-community/whisper-small-mlx-8bit",
        "mlx-community/whisper-small-mlx-4bit",
        "mlx-community/whisper-small.en-mlx",
        "mlx-community/whisper-small.en-mlx-q4",
        "mlx-community/whisper-small.en-mlx-4bit",
        "mlx-community/whisper-small.en-mlx-8bit",
        # "mlx-community/whisper-small.en-mlx-fp32",
    ],
    "base": [
        "mlx-community/whisper-base-mlx",
        "mlx-community/whisper-base-mlx-q4",
        # "mlx-community/whisper-base-mlx-fp32",
        "mlx-community/whisper-base-mlx-8bit",
        # "mlx-community/whisper-base-mlx-2bit",
        "mlx-community/whisper-base-mlx-4bit",
        "mlx-community/whisper-base.en-mlx",
        "mlx-community/whisper-base.en-mlx-q4",
        "mlx-community/whisper-base.en-mlx-4bit",
        "mlx-community/whisper-base.en-mlx-8bit",
        # "mlx-community/whisper-base.en-mlx-fp32",
    ],
    "medium": [
        "mlx-community/whisper-medium-mlx-8bit",
        # "mlx-community/whisper-medium-mlx-fp32",
        "mlx-community/whisper-medium-mlx-q4",
        "mlx-community/whisper-medium.en-mlx-8bit",
        "mlx-community/whisper-medium.en-mlx-4bit",
        # "mlx-community/whisper-medium.en-mlx-fp32",
    ],
    "large": [
        "mlx-community/whisper-large-mlx",
        "mlx-community/whisper-large-mlx-4bit",
        "mlx-community/whisper-large-mlx-8bit",
        "mlx-community/whisper-large-v1-mlx",
        "mlx-community/whisper-large-v1-mlx-4bit",
        "mlx-community/whisper-large-v1-mlx-8bit",
        "mlx-community/whisper-large-v2-mlx-4bit",
        "mlx-community/whisper-large-v2-mlx-8bit",
        # "mlx-community/whisper-large-v2-mlx-fp32",
        "mlx-community/whisper-large-v3-mlx",
        "mlx-community/whisper-large-v3-turbo",
    ],
    "turbo": [
        "mlx-community/whisper-turbo",
    ],
}

# Default mapping (current behavior)
DEFAULT_MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "medium": "mlx-community/whisper-medium-mlx-8bit",
    "large": "mlx-community/whisper-large-mlx-8bit",
    "turbo": "mlx-community/whisper-turbo",
}


class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom formatter to avoid parameter duplication in help text."""

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)

        # Format as "-s, --summary" instead of "-s {text,markdown,csv,html}, --summary {text,markdown,csv,html}"
        default = self._get_default_metavar_for_optional(action)
        args_string = self._format_args(action, default)
        return ", ".join(action.option_strings) + " " + args_string


def create_cpu_whisper_options(
    model_size: str = "turbo", language: Optional[str] = None
):
    """Create native Whisper options forced to CPU."""
    # For auto-detection, pass empty string instead of defaulting to "en"
    whisper_language = language if language is not None else ""

    return InlineAsrNativeWhisperOptions(
        repo_id=model_size,
        inference_framework=InferenceAsrFramework.WHISPER,
        language=whisper_language,
        verbose=True,
        timestamps=True,
        word_timestamps=True,
        temperature=0.0,
        max_new_tokens=256,
        max_time_chunk=30.0,
    )


def create_mlx_whisper_options(repo_id: str, language: Optional[str] = None):
    """Create MLX Whisper options for Apple Silicon."""
    # For auto-detection, pass empty string - the MLX model will convert it to None
    whisper_language = language if language is not None else ""

    return InlineAsrMlxWhisperOptions(
        repo_id=repo_id,
        inference_framework=InferenceAsrFramework.MLX,
        language=whisper_language,
        task="transcribe",
        word_timestamps=True,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )


def parse_mlx_model_override(override_list):
    """Parse --mlx-model-map arguments like base=repo"""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid format: {item}. Expected MODEL=REPO")
        model, repo = item.split("=", 1)
        if model not in DEFAULT_MLX_MODEL_MAP:
            raise ValueError(f"Unknown model: {model}")
        overrides[model] = repo
    return overrides


def get_mlx_models_for_test(args):
    """Determine which MLX models to test based on arguments"""
    mlx_models = {}

    for model_size in args.model:
        if args.mlx_all_variants and model_size in args.mlx_all_variants:
            # Use all variants
            mlx_models[model_size] = MLX_MODEL_REGISTRY[model_size]
        elif args.mlx_model_map and model_size in args.mlx_model_map:
            # Use override
            mlx_models[model_size] = [args.mlx_model_map[model_size]]
        else:
            # Use default
            mlx_models[model_size] = [DEFAULT_MLX_MODEL_MAP[model_size]]

    return mlx_models


def list_mlx_models(filter_model=None):
    """List available MLX models"""
    if filter_model:
        if filter_model not in MLX_MODEL_REGISTRY:
            print(f"Unknown model: {filter_model}")
            return
        print(f"MLX models for {filter_model}:")
        for repo in MLX_MODEL_REGISTRY[filter_model]:
            print(f"  {repo}")
    else:
        print("All available MLX models:")
        for model, repos in MLX_MODEL_REGISTRY.items():
            print(f"\n{model.upper()}:")
            for repo in repos:
                print(f"  {repo}")


def show_model_map(custom_map=None):
    """Show the current model mapping"""
    map_to_show = custom_map or DEFAULT_MLX_MODEL_MAP
    print("Current MLX Model Mapping:")
    print("-" * 60)
    for model, repo in map_to_show.items():
        print(f"  {model:<10} → {repo}")


def format_summary_text(results: Dict, model_sizes: List[str]) -> str:
    """Format results as text table."""
    output = []
    output.append("PERFORMANCE COMPARISON SUMMARY")
    output.append("=" * 100)
    output.append(
        f"{'Model':<10} {'CPU Repo':<25} {'MLX Repo':<40} {'CPU (sec)':<12} {'MLX (sec)':<12} {'Speedup':<12} {'Status':<10}"
    )
    output.append("-" * 100)

    speedups = []

    for model_size in model_sizes:
        if model_size not in results:
            continue

        model_results = results[model_size]

        # Handle CPU results
        if "cpu" in model_results:
            cpu_duration = model_results["cpu"]["duration"]
            cpu_success = model_results["cpu"]["success"]
            cpu_repo = model_results["cpu"]["repo_id"]
        else:
            cpu_duration = 0
            cpu_success = False
            cpu_repo = "N/A"

        # Handle MLX results (may be multiple variants)
        mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]

        if mlx_keys:
            for mlx_key in mlx_keys:
                mlx_duration = model_results[mlx_key]["duration"]
                mlx_success = model_results[mlx_key]["success"]
                mlx_repo = model_results[mlx_key]["repo_id"]

                if cpu_success and mlx_success:
                    speedup = cpu_duration / mlx_duration
                    speedups.append(speedup)
                    status = "✅ Both OK"
                elif cpu_success:
                    speedup = float("inf")
                    status = "❌ MLX Failed"
                elif mlx_success:
                    speedup = 0
                    status = "❌ CPU Failed"
                else:
                    speedup = 0
                    status = "❌ Both Failed"

                output.append(
                    f"{model_size:<10} {cpu_repo:<25} {mlx_repo:<40} {cpu_duration:<12.2f} {mlx_duration:<12.2f} {speedup:<12.2f}x {status:<10}"
                )
        else:
            # No MLX results
            output.append(
                f"{model_size:<10} {cpu_repo:<25} {'N/A':<40} {cpu_duration:<12.2f} {'N/A':<12} {'N/A':<12} {'❌ No MLX':<10}"
            )

    # Add speedup statistics
    if speedups:
        output.append("-" * 100)
        output.append("\nSPEEDUP STATISTICS")
        output.append("-" * 100)
        output.append(
            f"{'Metric':<10} {'CPU Repo':<25} {'MLX Repo':<40} {'CPU (sec)':<12} {'MLX (sec)':<12} {'Speedup':<12}"
        )
        output.append("-" * 100)

        # Calculate statistics
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        # Find corresponding models for min and max
        min_model = None
        max_model = None
        for model_size in model_sizes:
            if model_size not in results:
                continue
            model_results = results[model_size]
            mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]
            for mlx_key in mlx_keys:
                if (
                    model_results["cpu"]["success"]
                    and model_results[mlx_key]["success"]
                ):
                    speedup = (
                        model_results["cpu"]["duration"]
                        / model_results[mlx_key]["duration"]
                    )
                    if speedup == min_speedup:
                        min_model = (model_size, mlx_key)
                    if speedup == max_speedup:
                        max_model = (model_size, mlx_key)

        # Add statistics rows
        output.append(
            f"{'AVERAGE':<10} {'All Models':<25} {'All Models':<40} {'':<12} {'':<12} {avg_speedup:<12.2f}x"
        )
        if min_model:
            model_size, mlx_key = min_model
            min_results = results[model_size]
            output.append(
                f"{'MIN':<10} {min_results['cpu']['repo_id']:<25} {min_results[mlx_key]['repo_id']:<40} {min_results['cpu']['duration']:<12.2f} {min_results[mlx_key]['duration']:<12.2f} {min_speedup:<12.2f}x"
            )
        if max_model:
            model_size, mlx_key = max_model
            max_results = results[model_size]
            output.append(
                f"{'MAX':<10} {max_results['cpu']['repo_id']:<25} {max_results[mlx_key]['repo_id']:<40} {max_results['cpu']['duration']:<12.2f} {max_results[mlx_key]['duration']:<12.2f} {max_speedup:<12.2f}x"
            )

    return "\n".join(output)


def format_summary_markdown(results: Dict, model_sizes: List[str]) -> str:
    """Format results as markdown table."""
    output = []
    output.append("# Performance Comparison Summary")
    output.append("")
    output.append(
        "| Model | CPU Repository | MLX Repository | CPU (sec) | MLX (sec) | Speedup | Status |"
    )
    output.append(
        "|-------|----------------|----------------|-----------|-----------|---------|--------|"
    )

    speedups = []

    for model_size in model_sizes:
        if model_size not in results:
            continue

        model_results = results[model_size]

        # Handle CPU results
        if "cpu" in model_results:
            cpu_duration = model_results["cpu"]["duration"]
            cpu_success = model_results["cpu"]["success"]
            cpu_repo = model_results["cpu"]["repo_id"]
        else:
            cpu_duration = 0
            cpu_success = False
            cpu_repo = "N/A"

        # Handle MLX results (may be multiple variants)
        mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]

        if mlx_keys:
            for mlx_key in mlx_keys:
                mlx_duration = model_results[mlx_key]["duration"]
                mlx_success = model_results[mlx_key]["success"]
                mlx_repo = model_results[mlx_key]["repo_id"]

                if cpu_success and mlx_success:
                    speedup = cpu_duration / mlx_duration
                    speedups.append(speedup)
                    status = "✅ Both OK"
                elif cpu_success:
                    speedup = "∞"
                    status = "❌ MLX Failed"
                elif mlx_success:
                    speedup = "0"
                    status = "❌ CPU Failed"
                else:
                    speedup = "0"
                    status = "❌ Both Failed"

                output.append(
                    f"| {model_size} | `{cpu_repo}` | `{mlx_repo}` | {cpu_duration:.2f} | {mlx_duration:.2f} | {speedup:.2f}x | {status} |"
                )
        else:
            # No MLX results
            output.append(
                f"| {model_size} | `{cpu_repo}` | N/A | {cpu_duration:.2f} | N/A | N/A | ❌ No MLX |"
            )

    # Add speedup statistics
    if speedups:
        output.append("")
        output.append("## Speedup Statistics")
        output.append("")
        output.append(
            "| Metric | CPU Repository | MLX Repository | CPU (sec) | MLX (sec) | Speedup |"
        )
        output.append(
            "|--------|---------------|----------------|-----------|-----------|---------|"
        )

        # Calculate statistics
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        # Find corresponding models for min and max
        min_model = None
        max_model = None
        for model_size in model_sizes:
            if model_size not in results:
                continue
            model_results = results[model_size]
            mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]
            for mlx_key in mlx_keys:
                if (
                    model_results["cpu"]["success"]
                    and model_results[mlx_key]["success"]
                ):
                    speedup = (
                        model_results["cpu"]["duration"]
                        / model_results[mlx_key]["duration"]
                    )
                    if speedup == min_speedup:
                        min_model = (model_size, mlx_key)
                    if speedup == max_speedup:
                        max_model = (model_size, mlx_key)

        # Add statistics rows
        output.append(f"| Average | All Models | All Models | | | {avg_speedup:.2f}x |")
        if min_model:
            model_size, mlx_key = min_model
            min_results = results[model_size]
            output.append(
                f"| Min | `{min_results['cpu']['repo_id']}` | `{min_results[mlx_key]['repo_id']}` | {min_results['cpu']['duration']:.2f} | {min_results[mlx_key]['duration']:.2f} | {min_speedup:.2f}x |"
            )
        if max_model:
            model_size, mlx_key = max_model
            max_results = results[model_size]
            output.append(
                f"| Max | `{max_results['cpu']['repo_id']}` | `{max_results[mlx_key]['repo_id']}` | {max_results['cpu']['duration']:.2f} | {max_results[mlx_key]['duration']:.2f} | {max_speedup:.2f}x |"
            )

    return "\n".join(output)


def format_summary_csv(results: Dict, model_sizes: List[str]) -> str:
    """Format results as CSV."""
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "Model",
            "CPU_Repository",
            "MLX_Repository",
            "CPU_Seconds",
            "MLX_Seconds",
            "Speedup",
            "Status",
        ]
    )

    speedups = []

    for model_size in model_sizes:
        if model_size not in results:
            continue

        model_results = results[model_size]

        # Handle CPU results
        if "cpu" in model_results:
            cpu_duration = model_results["cpu"]["duration"]
            cpu_success = model_results["cpu"]["success"]
            cpu_repo = model_results["cpu"]["repo_id"]
        else:
            cpu_duration = 0
            cpu_success = False
            cpu_repo = "N/A"

        # Handle MLX results (may be multiple variants)
        mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]

        if mlx_keys:
            for mlx_key in mlx_keys:
                mlx_duration = model_results[mlx_key]["duration"]
                mlx_success = model_results[mlx_key]["success"]
                mlx_repo = model_results[mlx_key]["repo_id"]

                if cpu_success and mlx_success:
                    speedup = cpu_duration / mlx_duration
                    speedups.append(speedup)
                    status = "Both OK"
                elif cpu_success:
                    speedup = "inf"
                    status = "MLX Failed"
                elif mlx_success:
                    speedup = "0"
                    status = "CPU Failed"
                else:
                    speedup = "0"
                    status = "Both Failed"

                writer.writerow(
                    [
                        model_size,
                        cpu_repo,
                        mlx_repo,
                        f"{cpu_duration:.2f}",
                        f"{mlx_duration:.2f}",
                        f"{speedup:.2f}x",
                        status,
                    ]
                )
        else:
            # No MLX results
            writer.writerow(
                [
                    model_size,
                    cpu_repo,
                    "N/A",
                    f"{cpu_duration:.2f}",
                    "N/A",
                    "N/A",
                    "No MLX",
                ]
            )

    # Add speedup statistics
    if speedups:
        writer.writerow([])  # Empty row
        writer.writerow(
            [
                "Metric",
                "CPU_Repository",
                "MLX_Repository",
                "CPU_Seconds",
                "MLX_Seconds",
                "Speedup",
            ]
        )

        # Calculate statistics
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        # Find corresponding models for min and max
        min_model = None
        max_model = None
        for model_size in model_sizes:
            if model_size not in results:
                continue
            model_results = results[model_size]
            mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]
            for mlx_key in mlx_keys:
                if (
                    model_results["cpu"]["success"]
                    and model_results[mlx_key]["success"]
                ):
                    speedup = (
                        model_results["cpu"]["duration"]
                        / model_results[mlx_key]["duration"]
                    )
                    if speedup == min_speedup:
                        min_model = (model_size, mlx_key)
                    if speedup == max_speedup:
                        max_model = (model_size, mlx_key)

        # Add statistics rows
        writer.writerow(
            ["Average", "All Models", "All Models", "", "", f"{avg_speedup:.2f}x"]
        )
        if min_model:
            model_size, mlx_key = min_model
            min_results = results[model_size]
            writer.writerow(
                [
                    "Min",
                    min_results["cpu"]["repo_id"],
                    min_results[mlx_key]["repo_id"],
                    f"{min_results['cpu']['duration']:.2f}",
                    f"{min_results[mlx_key]['duration']:.2f}",
                    f"{min_speedup:.2f}x",
                ]
            )
        if max_model:
            model_size, mlx_key = max_model
            max_results = results[model_size]
            writer.writerow(
                [
                    "Max",
                    max_results["cpu"]["repo_id"],
                    max_results[mlx_key]["repo_id"],
                    f"{max_results['cpu']['duration']:.2f}",
                    f"{max_results[mlx_key]['duration']:.2f}",
                    f"{max_speedup:.2f}x",
                ]
            )

    return output.getvalue()


def format_summary_html(results: Dict, model_sizes: List[str]) -> str:
    """Format results as HTML table."""
    output = []
    output.append("<h1>Performance Comparison Summary</h1>")
    output.append("<table border='1' style='border-collapse: collapse;'>")
    output.append(
        "<tr><th>Model</th><th>CPU Repository</th><th>MLX Repository</th><th>CPU (sec)</th><th>MLX (sec)</th><th>Speedup</th><th>Status</th></tr>"
    )

    speedups = []

    for model_size in model_sizes:
        if model_size not in results:
            continue

        model_results = results[model_size]

        # Handle CPU results
        if "cpu" in model_results:
            cpu_duration = model_results["cpu"]["duration"]
            cpu_success = model_results["cpu"]["success"]
            cpu_repo = model_results["cpu"]["repo_id"]
        else:
            cpu_duration = 0
            cpu_success = False
            cpu_repo = "N/A"

        # Handle MLX results (may be multiple variants)
        mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]

        if mlx_keys:
            for mlx_key in mlx_keys:
                mlx_duration = model_results[mlx_key]["duration"]
                mlx_success = model_results[mlx_key]["success"]
                mlx_repo = model_results[mlx_key]["repo_id"]

                if cpu_success and mlx_success:
                    speedup = cpu_duration / mlx_duration
                    speedups.append(speedup)
                    status = "✅ Both OK"
                elif cpu_success:
                    speedup = "∞"
                    status = "❌ MLX Failed"
                elif mlx_success:
                    speedup = "0"
                    status = "❌ CPU Failed"
                else:
                    speedup = "0"
                    status = "❌ Both Failed"

                output.append(
                    f"<tr><td>{model_size}</td><td><code>{cpu_repo}</code></td><td><code>{mlx_repo}</code></td><td>{cpu_duration:.2f}</td><td>{mlx_duration:.2f}</td><td>{speedup:.2f}x</td><td>{status}</td></tr>"
                )
        else:
            # No MLX results
            output.append(
                f"<tr><td>{model_size}</td><td><code>{cpu_repo}</code></td><td>N/A</td><td>{cpu_duration:.2f}</td><td>N/A</td><td>N/A</td><td>❌ No MLX</td></tr>"
            )

    output.append("</table>")

    # Add speedup statistics
    if speedups:
        output.append("<h2>Speedup Statistics</h2>")
        output.append("<table border='1' style='border-collapse: collapse;'>")
        output.append(
            "<tr><th>Metric</th><th>CPU Repository</th><th>MLX Repository</th><th>CPU (sec)</th><th>MLX (sec)</th><th>Speedup</th></tr>"
        )

        # Calculate statistics
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)

        # Find corresponding models for min and max
        min_model = None
        max_model = None
        for model_size in model_sizes:
            if model_size not in results:
                continue
            model_results = results[model_size]
            mlx_keys = [k for k in model_results.keys() if k.startswith("mlx")]
            for mlx_key in mlx_keys:
                if (
                    model_results["cpu"]["success"]
                    and model_results[mlx_key]["success"]
                ):
                    speedup = (
                        model_results["cpu"]["duration"]
                        / model_results[mlx_key]["duration"]
                    )
                    if speedup == min_speedup:
                        min_model = (model_size, mlx_key)
                    if speedup == max_speedup:
                        max_model = (model_size, mlx_key)

        # Add statistics rows
        output.append(
            f"<tr><td>Average</td><td>All Models</td><td>All Models</td><td></td><td></td><td>{avg_speedup:.2f}x</td></tr>"
        )
        if min_model:
            model_size, mlx_key = min_model
            min_results = results[model_size]
            output.append(
                f"<tr><td>Min</td><td><code>{min_results['cpu']['repo_id']}</code></td><td><code>{min_results[mlx_key]['repo_id']}</code></td><td>{min_results['cpu']['duration']:.2f}</td><td>{min_results[mlx_key]['duration']:.2f}</td><td>{min_speedup:.2f}x</td></tr>"
            )
        if max_model:
            model_size, mlx_key = max_model
            max_results = results[model_size]
            output.append(
                f"<tr><td>Max</td><td><code>{max_results['cpu']['repo_id']}</code></td><td><code>{max_results[mlx_key]['repo_id']}</code></td><td>{max_results['cpu']['duration']:.2f}</td><td>{max_results[mlx_key]['duration']:.2f}</td><td>{max_speedup:.2f}x</td></tr>"
            )
        output.append("</table>")

    return "\n".join(output)


def run_transcription_test(
    audio_file: Path, asr_options, device: AcceleratorDevice, test_name: str
):
    """Run a single transcription test and return timing results."""
    print(f"\n{'=' * 60}")
    print(f"Running {test_name}")
    print(f"Device: {device}")
    print(f"Model: {asr_options.repo_id}")
    print(f"Framework: {asr_options.inference_framework}")
    print(f"{'=' * 60}")

    # Create pipeline options
    pipeline_options = AsrPipelineOptions(
        accelerator_options=AcceleratorOptions(device=device),
        asr_options=asr_options,
    )

    # Create document converter
    converter = DocumentConverter(
        format_options={
            InputFormat.AUDIO: AudioFormatOption(
                pipeline_cls=AsrPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )

    # Run transcription with timing
    start_time = time.time()
    try:
        result = converter.convert(audio_file)
        end_time = time.time()

        duration = end_time - start_time

        if result.status.value == "success":
            # Extract text for verification
            text_content = []
            for item in result.document.texts:
                text_content.append(item.text)

            print(f"✅ Success! Duration: {duration:.2f} seconds")
            print(f"Transcribed text: {''.join(text_content)[:200]}...")
            return duration, True, asr_options.repo_id
        else:
            print(f"❌ Failed! Status: {result.status}")
            return duration, False, asr_options.repo_id

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Error: {e}")
        return duration, False, asr_options.repo_id

    finally:
        # Clean up MLX resources after each MLX test to prevent semaphore leaks
        if device == AcceleratorDevice.MPS:
            cleanup_mlx_resources()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Performance comparison between CPU and MLX Whisper on Apple Silicon. Tests Whisper models with customizable device selection and output formats.",
        formatter_class=CustomHelpFormatter,
        epilog="""
Examples:

# Test all models with default settings (CPU + MLX, text output)
python asr_pipeline_performance_comparison.py

# Test specific models with markdown output
python asr_pipeline_performance_comparison.py -m tiny base turbo -f markdown

# Test all models with multiple output formats
python asr_pipeline_performance_comparison.py -m all -f html markdown csv

# Test only CPU performance with CSV output to custom directory
python asr_pipeline_performance_comparison.py -d cpu -f csv -o ./reports

# Test with custom audio file and language detection
python asr_pipeline_performance_comparison.py --audio /path/to/audio.wav -l None

# Test CPU and CUDA (if available) with HTML output
python asr_pipeline_performance_comparison.py -d cpu cuda -f html

# Output all formats with custom basename
python asr_pipeline_performance_comparison.py -f all --output-basename my_report

# Override specific MLX models
python asr_pipeline_performance_comparison.py -m base --mlx-model-map base=mlx-community/whisper-base.en-mlx-q4

# Test all MLX variants for small model
python asr_pipeline_performance_comparison.py -m small --mlx-all-variants small

# Test multiple models with different MLX variants
python asr_pipeline_performance_comparison.py -m tiny base \\
  --mlx-model-map tiny=mlx-community/whisper-tiny-mlx-8bit \\
                   base=mlx-community/whisper-base.en-mlx-q4

# List all available MLX models
python asr_pipeline_performance_comparison.py --list-mlx-models

# List MLX models for specific base model
python asr_pipeline_performance_comparison.py --list-mlx-models small

# Show current model mapping
python asr_pipeline_performance_comparison.py --show-model-map

# Test all variants of small and base models
python asr_pipeline_performance_comparison.py \\
  --mlx-all-variants small base -s markdown
        """,
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file for testing (default: tests/data/audio/sample_10s.mp3)",
    )

    parser.add_argument(
        "-f",
        "--format",
        nargs="+",
        choices=["text", "markdown", "csv", "html", "all"],
        default=["text"],
        help="Output format(s) for summary reports (default: text). Use 'all' to output all formats.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to output summary reports (default: ./output)",
    )

    parser.add_argument(
        "--output-basename",
        type=str,
        default=None,
        help="Override the summary report basename (default: script basename)",
    )

    parser.add_argument(
        "-d",
        "--device",
        nargs="+",
        choices=["cpu", "mlx", "cuda"],
        default=["cpu", "mlx"],
        help="Devices to test (must choose at least two, default: cpu mlx)",
    )

    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default=None,
        help="Language spoken in the audio, specify 'None' to auto-detect (default: None)",
    )

    parser.add_argument(
        "-m",
        "--model",
        nargs="+",
        choices=["tiny", "small", "base", "medium", "large", "turbo", "all"],
        default=["tiny", "base", "turbo"],
        help="Models to test. Use 'all' to test all available models (default: tiny base turbo)",
    )

    parser.add_argument(
        "--mlx-model-map",
        nargs="+",
        help="Override specific MLX models (e.g., --mlx-model-map base=mlx-community/whisper-base.en-mlx-q4)",
    )

    parser.add_argument(
        "--mlx-all-variants",
        nargs="+",
        choices=["tiny", "small", "base", "medium", "large", "turbo"],
        help="Test all MLX variants for specified models (e.g., --mlx-all-variants small base)",
    )

    parser.add_argument(
        "--list-mlx-models",
        nargs="?",
        choices=["tiny", "small", "base", "medium", "large", "turbo", "all"],
        const="all",
        help="List all available MLX models (optionally filter by base model)",
    )

    parser.add_argument(
        "--show-model-map",
        action="store_true",
        help="Show the current default model mapping",
    )

    args = parser.parse_args()

    # Validate device selection
    if len(args.device) < 2:
        parser.error("Must specify at least two devices for comparison")

    # Convert language string to None if specified as "None"
    if args.language and args.language.lower() == "none":
        args.language = None

    # Handle "all" option for models
    if "all" in args.model:
        args.model = ["tiny", "small", "base", "medium", "large", "turbo"]

    # Handle special commands
    if args.list_mlx_models:
        if args.list_mlx_models == "all":
            list_mlx_models()
        else:
            list_mlx_models(args.list_mlx_models)
        sys.exit(0)

    # Parse MLX model overrides
    if args.mlx_model_map:
        try:
            args.mlx_model_map = parse_mlx_model_override(args.mlx_model_map)
        except ValueError as e:
            parser.error(str(e))

    if args.show_model_map:
        # Create custom map if overrides are provided
        custom_map = DEFAULT_MLX_MODEL_MAP.copy()
        if args.mlx_model_map:
            custom_map.update(args.mlx_model_map)
        show_model_map(custom_map)
        sys.exit(0)

    return args


def write_output_files(results: Dict, model_sizes: List[str], args, script_name: str):
    """Write output files based on format options."""
    # Handle "all" format
    if "all" in args.format:
        formats_to_write = ["text", "markdown", "csv", "html"]
    else:
        formats_to_write = args.format

    # Determine basename
    if args.output_basename:
        basename = args.output_basename
    else:
        basename = script_name

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Write files for each format (except text which goes to stdout)
    for format_type in formats_to_write:
        if format_type == "text":
            continue  # Text goes to stdout, not to file

        ft = format_type if format_type != "markdown" else "md"
        # Generate filename
        filename = f"{basename}_{timestamp}.{ft}"
        filepath = output_dir / filename

        # Generate content
        if format_type == "markdown":
            content = format_summary_markdown(results, model_sizes)
        elif format_type == "csv":
            content = format_summary_csv(results, model_sizes)
        elif format_type == "html":
            content = format_summary_html(results, model_sizes)
        else:
            continue

        # Write file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"📄 {format_type.upper()} summary written to: {filepath}")


def main():
    """Run performance comparison between CPU and MLX Whisper."""
    args = parse_args()

    # Check device availability
    try:
        import torch

        has_mps = torch.backends.mps.is_built() and torch.backends.mps.is_available()
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_mps = False
        has_cuda = False

    try:
        import mlx_whisper

        has_mlx_whisper = True
    except ImportError:
        has_mlx_whisper = False

    print("ASR Pipeline Performance Comparison")
    print("=" * 50)
    print(f"Testing models: {', '.join(args.model)}")
    print(f"Testing devices: {', '.join(args.device)}")
    print(f"Output formats: {', '.join(args.format)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Language: {args.language or 'auto-detect'}")
    print(f"Apple Silicon (MPS) available: {has_mps}")
    print(f"CUDA available: {has_cuda}")
    print(f"MLX Whisper available: {has_mlx_whisper}")

    # Validate device availability
    available_devices = []
    if "cpu" in args.device:
        available_devices.append("cpu")
    if "mlx" in args.device and has_mps and has_mlx_whisper:
        available_devices.append("mlx")
    elif "mlx" in args.device:
        print("⚠️  MLX Whisper not available - skipping MLX tests")
        print("   Install with: pip install mlx-whisper")
        print("   Or: uv sync --extra asr")
    if "cuda" in args.device and has_cuda:
        available_devices.append("cuda")
    elif "cuda" in args.device:
        print("⚠️  CUDA not available - skipping CUDA tests")

    if len(available_devices) < 2:
        print("❌ Not enough available devices for comparison")
        print("   Available devices:", available_devices)
        sys.exit(1)

    # Determine audio file path
    if args.audio:
        audio_file = Path(args.audio)
        if not audio_file.is_absolute():
            audio_file = Path(__file__).parent.parent.parent / audio_file
    else:
        audio_file = (
            Path(__file__).parent.parent.parent
            / "tests"
            / "data"
            / "audio"
            / "sample_10s.mp3"
        )

    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("   Please check the path and try again.")
        sys.exit(1)

    print(f"Using test audio: {audio_file}")
    print(f"File size: {audio_file.stat().st_size / 1024:.1f} KB")

    # Get MLX models to test
    mlx_models_to_test = get_mlx_models_for_test(args)

    # Run tests
    results = {}

    for model_size in args.model:
        print(f"\n{'#' * 80}")
        print(f"Testing model size: {model_size}")
        print(f"{'#' * 80}")

        model_results = {}

        # Test CPU
        if "cpu" in available_devices:
            cpu_options = create_cpu_whisper_options(model_size, args.language)
            cpu_duration, cpu_success, cpu_repo = run_transcription_test(
                audio_file,
                cpu_options,
                AcceleratorDevice.CPU,
                f"Native Whisper {model_size} (CPU)",
            )
            model_results["cpu"] = {
                "duration": cpu_duration,
                "success": cpu_success,
                "repo_id": cpu_repo,
            }

        # Test MLX variants
        if "mlx" in available_devices:
            mlx_repos = mlx_models_to_test[model_size]
            for i, mlx_repo in enumerate(mlx_repos):
                test_name = f"MLX Whisper {model_size} (MPS)"
                if len(mlx_repos) > 1:
                    test_name += f" [{i + 1}/{len(mlx_repos)}]"

                mlx_options = create_mlx_whisper_options(mlx_repo, args.language)
                mlx_duration, mlx_success, mlx_repo_id = run_transcription_test(
                    audio_file,
                    mlx_options,
                    AcceleratorDevice.MPS,
                    test_name,
                )

                # Store results with unique key for multiple variants
                if len(mlx_repos) > 1:
                    key = f"mlx_{i}"
                else:
                    key = "mlx"

                model_results[key] = {
                    "duration": mlx_duration,
                    "success": mlx_success,
                    "repo_id": mlx_repo_id,
                }

        # Test CUDA
        if "cuda" in available_devices:
            cuda_options = create_cpu_whisper_options(model_size, args.language)
            cuda_duration, cuda_success, cuda_repo = run_transcription_test(
                audio_file,
                cuda_options,
                AcceleratorDevice.CUDA,
                f"Native Whisper {model_size} (CUDA)",
            )
            model_results["cuda"] = {
                "duration": cuda_duration,
                "success": cuda_success,
                "repo_id": cuda_repo,
            }

        results[model_size] = model_results

        # Clean up MLX resources after each model size to prevent accumulation
        if "mlx" in available_devices:
            print(f"\n🧹 Cleaning up MLX resources after {model_size} model tests...")
            # Use aggressive cleanup for comprehensive tests
            if len(args.model) > 3 or any(
                size in args.model for size in ["large", "turbo"]
            ):
                aggressive_cleanup_mlx_resources()
            else:
                cleanup_mlx_resources()

    # Generate and display summary
    print(f"\n{'#' * 80}")

    # Always generate and display text summary to stdout
    text_summary = format_summary_text(results, args.model)
    print(f"\n{text_summary}\n\n")

    # Write additional format files if requested
    script_name = Path(__file__).stem  # Get script name without extension
    write_output_files(results, args.model, args, script_name)

    # Final cleanup to ensure all MLX resources are released
    if "mlx" in args.device:
        print("\n🧹 Performing final MLX resource cleanup...")
        # Use aggressive cleanup for comprehensive tests
        if len(args.model) > 3 or any(
            size in args.model for size in ["large", "turbo"]
        ):
            aggressive_cleanup_mlx_resources()
        else:
            # Multiple cleanup passes to ensure thorough resource release
            for i in range(5):
                cleanup_mlx_resources()
                if i < 4:  # Don't sleep after the last cleanup
                    time.sleep(0.5)


if __name__ == "__main__":
    main()
