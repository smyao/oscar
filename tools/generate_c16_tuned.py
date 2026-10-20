"""Archive #145/#164, D.4: targeted C16 variants from the preserved reader.

FIA/history only; never restore full history. Transformations retain FP32
online state, guards and original cross-core ownership. Runtime/measurement
qualification is separate from generating a candidate.
"""
from pathlib import Path
import argparse
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "csrc/kernels/attention_cv_aligned_cluster16_control.cpp"
OUTPUT = ROOT / "csrc/kernels/attention_cv_aligned_cluster16.cpp"


def generate(softmax=True, cast=True):
    text = SOURCE.read_text().replace("attention_cv_aligned_cluster16_control", "attention_cv_aligned_cluster16")
    identity = hashlib.sha256(text.encode()).hexdigest()
    if softmax:
        from tools.c16_softmax_transform import transform
        text = transform(text)
    if cast:
        from tools.c16_cast_transform import transform
        text = transform(text)
    text = text.replace("attention_cv_aligned_cluster16", "attention_cv_aligned_cluster16_tuned")
    return (f"// Generated from preserved C16 SHA256 {identity}\n"
            f"// Timer-directed options: softmax={int(softmax)} cast={int(cast)}.\n"
            "// D.4: fused history/window FIA; old full restoration cost6.5s.\n"
            "// This keeps bounded tiles, cache bytes and FP32 online math;\n"
            "// no full-history arena or protocol shortcuts. Speed is unmeasured.\n" + text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--only", choices=("softmax", "cast", "both"), default="both")
    args = parser.parse_args()
    if args.only != "both" and args.output == OUTPUT:
        raise ValueError("single-arm research requires an explicit separate output path")
    text = generate(args.only != "cast", args.only != "softmax")
    if args.output == OUTPUT:
        # Keep the production ABI/source name consumed by the unchanged timer.
        text = text.replace("attention_cv_aligned_cluster16_tuned", "attention_cv_aligned_cluster16")
    if args.check:
        if args.output.read_text() != text:
            raise ValueError("tuned C16 source drift")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
