import os
import sys
import time
import argparse
import threading
from queue import Queue
from pathlib import Path
from faster_whisper import WhisperModel
from faster_whisper import BatchedInferencePipeline

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

# Queue for producer → consumer pipeline
job_queue = Queue(maxsize=8)


def atomic_write_text(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def format_hms(seconds: float):

    seconds = max(0.0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    if h:
        return f"{h}:{m:02d}:{s:02d}"

    return f"{m:02d}:{s:02d}"


def collect_audio_files(p: Path, recursive: bool):

    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return [p]

    if p.is_dir():

        it = p.rglob("*") if recursive else p.glob("*")

        return sorted(
            [
                f
                for f in it
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS
            ]
        )

    return []


def live_iter_segments(segments_iter, total_duration, label):

    seg_count = 0
    last_ui = 0
    t0 = time.time()

    print(f"-> {label}")

    for seg in segments_iter:

        seg_count += 1

        if total_duration and total_duration > 0:

            now = time.time()

            if now - last_ui >= 0.2:

                pct = min(
                    100.0,
                    (float(seg.end) / float(total_duration)) * 100.0,
                )

                elapsed = now - t0

                print(
                    f"\r   {pct:6.2f}%  {format_hms(seg.end):>8}/{format_hms(total_duration):>8}  "
                    f"segs:{seg_count:<5}  elapsed:{format_hms(elapsed)}",
                    end="",
                    flush=True,
                )

                last_ui = now

        yield seg

    if total_duration:

        elapsed = time.time() - t0

        print(
            f"\r   100.00%  {format_hms(total_duration):>8}/{format_hms(total_duration):>8}  "
            f"segs:{seg_count:<5}  elapsed:{format_hms(elapsed)}"
        )


def transcribe_with_pipeline(
    pipeline,
    audio_path,
    language,
    beam_size,
    batch_size,
    vad_filter,
    vad_min_silence_ms,
    chunk_length,
):

    segments_iter, info = pipeline.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        beam_size=beam_size,
        batch_size=batch_size,
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": vad_min_silence_ms
        }
        if vad_filter
        else None,
        chunk_length=chunk_length,
    )

    return segments_iter, info


# -----------------------------------
# Producer thread (CPU)
# -----------------------------------

def producer(files):

    for f in files:

        print(f"[CPU] preparing {f.name}")

        job_queue.put(f)

    # signal completion
    job_queue.put(None)


# -----------------------------------
# Consumer thread (GPU)
# -----------------------------------

def consumer(
    batched,
    out_dir,
    args,
    total_files
):

    ok = 0
    index = 1

    while True:

        audio_file = job_queue.get()

        if audio_file is None:
            break

        out_path = out_dir / f"{audio_file.stem}.txt"

        if out_path.exists() and not args.overwrite:

            print(f"[{index}/{total_files}] ⏭️ {audio_file.name} exists")

            ok += 1
            index += 1
            continue

        print(f"[{index}/{total_files}]")

        try:

            segments_iter, info = transcribe_with_pipeline(
                batched,
                audio_file,
                args.language,
                args.beam_size,
                args.batch_size,
                args.vad,
                args.vad_min_silence_ms,
                args.chunk_length,
            )

            total = getattr(info, "duration", None)

            parts = []

            for seg in live_iter_segments(
                segments_iter, total, audio_file.name
            ):
                parts.append(seg.text or "")

            transcript_text = "".join(parts).strip()

            if total:
                duration_str = format_hms(total)
                final_text = f"Audio Length: {duration_str}\n\n{transcript_text}"
            else:
                final_text = transcript_text

            atomic_write_text(out_path, final_text)

            print(f"✅ saved: {out_path.name}\n")

            ok += 1
            index += 1

        except Exception as e:

            print(f"❌ {audio_file.name}: {e}")
            sys.exit(1)


def main():

    ap = argparse.ArgumentParser(description="GPU-only Urdu transcription")

    ap.add_argument("input")
    ap.add_argument("-o", "--output_dir", default="data/transcripts")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    ap.add_argument("--device_index", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--language", default="ur")
    ap.add_argument("--beam_size", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--chunk_length", type=int, default=None)
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--vad_min_silence_ms", type=int, default=500)

    ap.add_argument(
        "--cache_dir",
        default=r"D:\AI_MODELS\whisper\faster_whisper_cache",
    )

    ap.add_argument("--local_only", action="store_true")

    args = ap.parse_args()

    if args.vad:
        try:
            import onnxruntime
        except:
            print("Install: pip install onnxruntime")
            sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper model...")

    model = WhisperModel(
        args.model,
        device="cuda",
        device_index=args.device_index,
        compute_type=args.compute_type,
        num_workers=args.num_workers,
        download_root=str(cache_dir),
        local_files_only=args.local_only,
    )

    batched = BatchedInferencePipeline(model=model)

    inp = Path(args.input)

    files = collect_audio_files(inp, recursive=args.recursive)

    if not files:
        print("No audio files found")
        sys.exit(1)

    print(f"Found {len(files)} files")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # Start producer and consumer threads
    producer_thread = threading.Thread(target=producer, args=(files,))
    consumer_thread = threading.Thread(
        target=consumer,
        args=(batched, out_dir, args, len(files)),
    )

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    elapsed = time.time() - start

    print(f"Done in {format_hms(elapsed)}")


if __name__ == "__main__":
    main()