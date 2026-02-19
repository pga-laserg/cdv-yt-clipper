import argparse
import json
import os
import sys
import traceback
import warnings

def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return (
        os.getenv("PYANNOTE_ACCESS_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
    )


def diarize(audio_file: str, token: str) -> list[dict]:
    # PyTorch 2.6+ defaults torch.load(weights_only=True), which breaks
    # some pyannote checkpoints. We trust pyannote HF checkpoints here.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    warnings.filterwarnings("ignore", category=UserWarning)
    from pyannote.audio import Pipeline

    # Official model id for the requested pipeline.
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token,
        )
    except TypeError:
        # Compatibility path for older pyannote/hf hub combinations.
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )

    diarization = pipeline(audio_file)
    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speaker diarization with pyannote 3.1")
    parser.add_argument("audio_file", help="Path to wav/mp3/m4a audio file")
    parser.add_argument(
        "--token",
        help="Hugging Face access token. If omitted, uses PYANNOTE_ACCESS_TOKEN, HUGGINGFACE_TOKEN, or HF_TOKEN.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: audio file not found: {args.audio_file}", file=sys.stderr)
        sys.exit(1)

    token = resolve_token(args.token)
    if not token:
        print(
            "Error: missing Hugging Face token. Set PYANNOTE_ACCESS_TOKEN (or HUGGINGFACE_TOKEN/HF_TOKEN), or pass --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        print("Loading pyannote/speaker-diarization-3.1...", file=sys.stderr)
        segments = diarize(args.audio_file, token)
        print(json.dumps(segments))
    except Exception as exc:
        print(f"Diarization error: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(
            "If this is a gated model error, accept terms for both "
            "'pyannote/speaker-diarization-3.1' and 'pyannote/segmentation-3.0' on Hugging Face.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
