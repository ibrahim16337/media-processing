from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.paths import (
    METADATA_BATCH_DIR,
    METADATA_EXCEL_IMPORT_DIR,
    METADATA_SINGLE_DIR,
    slugify,
)
from app.pipelines.metadata_generation_pipeline.metadata_excel_exporter import (
    export_metadata_excel,
)
from app.pipelines.metadata_generation_pipeline.ollama_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_RETRIES,
    DEFAULT_SLEEP_MS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    generate_metadata_from_prompt,
)
from app.pipelines.metadata_generation_pipeline.prompt_builder import (
    build_metadata_prompt,
)
from app.pipelines.metadata_generation_pipeline.response_parser import (
    build_result_row,
    parse_metadata_json,
)
from app.pipelines.metadata_generation_pipeline.transcript_sources import (
    load_single_transcript_file,
    load_transcript_folder,
    load_transcripts_from_excel,
)


SUPPORTED_SOURCE_TYPES = {"single_file", "folder", "excel"}


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _append_tsv_row(path: Path, values: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = "\t".join(_safe_string(v).replace("\t", " ").replace("\n", " ") for v in values) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _resolve_base_output_dir(source_type: str) -> Path:
    if source_type == "single_file":
        return METADATA_SINGLE_DIR
    if source_type == "folder":
        return METADATA_BATCH_DIR
    if source_type == "excel":
        return METADATA_EXCEL_IMPORT_DIR
    raise ValueError(f"Unsupported source type: {source_type}")


def _load_source_items(
    source_type: str,
    source_path: str | Path,
    transcript_column: str | None = None,
    filename_column: str | None = None,
    sheet_name: str | None = None,
) -> list[dict[str, Any]]:
    if source_type == "single_file":
        return load_single_transcript_file(source_path)

    if source_type == "folder":
        return load_transcript_folder(source_path)

    if source_type == "excel":
        if not transcript_column:
            raise ValueError("transcript_column is required when source_type='excel'.")

        return load_transcripts_from_excel(
            excel_path=source_path,
            transcript_column=transcript_column,
            filename_column=filename_column,
            sheet_name=sheet_name,
        )

    raise ValueError(f"Unsupported source type: {source_type}")


def _build_run_dir(
    source_type: str,
    source_path: str | Path,
    output_name: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = _resolve_base_output_dir(source_type)

    source_path_obj = Path(source_path)
    default_name = source_path_obj.stem if source_path_obj.is_file() else source_path_obj.name
    run_name = slugify(output_name or default_name or source_type)
    run_dir = base_dir / f"{_timestamp_str()}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_metadata_generation(
    source_type: str,
    source_path: str | Path,
    transcript_column: str | None = None,
    filename_column: str | None = None,
    sheet_name: str | None = None,
    output_name: str | None = None,
    output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    system_message: str = "ہمیشہ ان پٹ کی زبان کا احترام کریں اور اگر ان پٹ اردو میں ہو تو اردو میں واضح، درست اور باوقار انداز میں جواب دیں۔",
) -> dict[str, Any]:
    source_type = _safe_string(source_type)

    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"Unsupported source_type: {source_type}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_SOURCE_TYPES))}"
        )

    items = _load_source_items(
        source_type=source_type,
        source_path=source_path,
        transcript_column=transcript_column,
        filename_column=filename_column,
        sheet_name=sheet_name,
    )

    run_dir = _build_run_dir(
        source_type=source_type,
        source_path=source_path,
        output_name=output_name,
        output_dir=output_dir,
    )

    raw_dir = run_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ok_log = run_dir / "llm_ok.tsv"
    err_log = run_dir / "llm_err.tsv"
    excel_output = run_dir / "metadata_output.xlsx"

    ok_log.write_text("status\tindex\tfilename\tchars\tattempt\n", encoding="utf-8")
    err_log.write_text("status\tindex\tfilename\terror\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []

    if not items:
        export_metadata_excel(rows=[], output_file=excel_output)
        return {
            "ok": True,
            "source_type": source_type,
            "source_path": str(source_path),
            "run_dir": str(run_dir),
            "excel_output": str(excel_output),
            "ok_log": str(ok_log),
            "err_log": str(err_log),
            "total_items": 0,
            "success_count": 0,
            "error_count": 0,
            "rows": [],
            "errors": [],
        }

    total_items = len(items)

    for index, item in enumerate(items, start=1):
        filename = _safe_string(item.get("filename", f"item_{index:03d}"))
        transcript_text = item.get("transcript", "") or ""

        prompt = build_metadata_prompt(transcript_text)

        llm_result = generate_metadata_from_prompt(
            user_prompt=prompt,
            model=model,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            sleep_ms=sleep_ms,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
            seed=seed,
            system_message=system_message,
        )

        raw_response_text = llm_result.get("content", "") or ""
        raw_response_file = raw_dir / f"{index:03d}_{slugify(Path(filename).stem or filename)}_response.txt"
        _write_text(raw_response_file, raw_response_text)

        if not llm_result.get("ok", False):
            error_message = _safe_string(llm_result.get("error", "Unknown Ollama error"))
            _append_tsv_row(err_log, ["HTTPERR", index, filename, error_message])

            rows.append(
                {
                    "filename": filename,
                    "title": "",
                    "description": "",
                    "tags": "",
                    "hashtags": "",
                }
            )

            errors.append(
                {
                    "index": index,
                    "filename": filename,
                    "stage": "ollama_call",
                    "error": error_message,
                    "raw_response_file": str(raw_response_file),
                }
            )

            if sleep_ms > 0 and index < total_items:
                time.sleep(sleep_ms / 1000.0)
            continue

        parsed = parse_metadata_json(raw_response_text)
        row = build_result_row(filename=filename, parsed_result=parsed)
        rows.append(row)

        if parsed.get("ok", False):
            _append_tsv_row(
                ok_log,
                [
                    "OK",
                    index,
                    filename,
                    len(raw_response_text),
                    llm_result.get("attempt", ""),
                ],
            )
        else:
            parse_error = _safe_string(parsed.get("error", "Invalid metadata JSON"))
            _append_tsv_row(err_log, ["PARSEERR", index, filename, parse_error])

            errors.append(
                {
                    "index": index,
                    "filename": filename,
                    "stage": "response_parse",
                    "error": parse_error,
                    "raw_response_file": str(raw_response_file),
                }
            )

        if sleep_ms > 0 and index < total_items:
            time.sleep(sleep_ms / 1000.0)

    export_metadata_excel(rows=rows, output_file=excel_output)

    return {
        "ok": True,
        "source_type": source_type,
        "source_path": str(source_path),
        "run_dir": str(run_dir),
        "excel_output": str(excel_output),
        "ok_log": str(ok_log),
        "err_log": str(err_log),
        "raw_dir": str(raw_dir),
        "total_items": total_items,
        "success_count": total_items - len(errors),
        "error_count": len(errors),
        "rows": rows,
        "errors": errors,
        "config": {
            "model": model,
            "base_url": base_url,
            "timeout": timeout,
            "retries": retries,
            "sleep_ms": sleep_ms,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "seed": seed,
        },
    }