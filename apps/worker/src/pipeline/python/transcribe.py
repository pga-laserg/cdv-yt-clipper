import sys
import json
import os
import argparse
from faster_whisper import WhisperModel

def transcribe_audio(file_path, model_size, beam_size):
    # Reduce parallel tokenizer worker churn that can lead to leaked semaphore warnings.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Check if CUDA is available, otherwise allow CPU
    device = "cpu"
    compute_type = "int8"
    
    print(f"Loading Whisper model '{model_size}' on {device}...", file=sys.stderr)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"Transcribing {file_path} (Forcing Spanish)...", file=sys.stderr)
    # Force language to Spanish to avoid English hallucinations/translations
    segments, info = model.transcribe(
        file_path, 
        beam_size=beam_size,
        language="es", 
        task="transcribe"
    )

    print(f"Detected language '{info.language}' with probability {info.language_probability}", file=sys.stderr)

    result_segments = []
    for segment in segments:
        result_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        # Optional: Print progress to stderr
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text), file=sys.stderr)

    return result_segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using faster-whisper.")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--model", default="small", help="Whisper model size (e.g. base, small)")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding")
    args = parser.parse_args()

    audio_file = args.audio_file
    
    if not os.path.exists(audio_file):
        print(f"Error: File {audio_file} not found", file=sys.stderr)
        sys.exit(1)

    try:
        result = transcribe_audio(audio_file, args.model, args.beam_size)
        print(json.dumps(result))
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        sys.exit(1)
