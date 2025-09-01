"""Command line interface for roomkit."""

from __future__ import annotations

import argparse
import json

import numpy as np

from roomkit import __version__
from roomkit.audio.io import load_audio, save_audio
from roomkit.codec.frontend import CodecFrontend
from roomkit.denoise import denoise
from roomkit.dereverb import dereverb
from roomkit.diarization import diarize
from roomkit.room.rt60 import estimate_rt60
from roomkit.types import Segment
from roomkit.vad import detect_voice


def _cmd_denoise(args: argparse.Namespace) -> int:
    samples, rate = load_audio(args.input)
    enhanced = denoise(
        np.asarray(samples, dtype=np.float64),
        method=args.method,
        noise_frames=args.noise_frames,
    )
    save_audio(args.output, enhanced, rate)
    print(f"wrote {args.output}")
    return 0


def _cmd_dereverb(args: argparse.Namespace) -> int:
    samples, rate = load_audio(args.input)
    enhanced = dereverb(
        np.asarray(samples, dtype=np.float64),
        taps=args.taps,
        delay=args.delay,
        iterations=args.iterations,
    )
    save_audio(args.output, enhanced, rate)
    print(f"wrote {args.output}")
    return 0


def _segments_payload(segments: list[Segment]) -> list[dict[str, float | str]]:
    return [
        {"start": segment.start, "end": segment.end, "label": segment.label}
        for segment in segments
    ]


def _cmd_vad(args: argparse.Namespace) -> int:
    samples, rate = load_audio(args.input)
    segments = detect_voice(
        np.asarray(samples, dtype=np.float64),
        sample_rate=rate,
        threshold_db=args.threshold_db,
    )
    payload = json.dumps(_segments_payload(segments), indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")
        print(f"wrote {args.output}")
    else:
        print(payload)
    return 0


def _cmd_diarize(args: argparse.Namespace) -> int:
    samples, rate = load_audio(args.input)
    segments = diarize(
        np.asarray(samples, dtype=np.float64),
        num_speakers=args.num_speakers,
        sample_rate=rate,
    )
    payload = json.dumps(_segments_payload(segments), indent=2)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")
        print(f"wrote {args.output}")
    else:
        print(payload)
    return 0


def _cmd_rt60(args: argparse.Namespace) -> int:
    samples, rate = load_audio(args.input)
    rt60 = estimate_rt60(np.asarray(samples, dtype=np.float64), rate)
    print(f"{rt60:.3f}")
    return 0


def _cmd_codec_encode(args: argparse.Namespace) -> int:
    samples, rate = load_audio(args.input)
    frontend = CodecFrontend(
        n_codebooks=args.codebooks, codebook_size=args.codebook_size, sample_rate=rate
    )
    audio = np.asarray(samples, dtype=np.float64)
    frontend.fit(audio)
    tokens = frontend.encode(audio)
    assert frontend._codebooks is not None  # noqa: SLF001
    np.savez(
        args.output,
        tokens=tokens,
        codebooks=np.stack(frontend._codebooks),  # noqa: SLF001
        sample_rate=np.int64(rate),
    )
    print(f"wrote {args.output}")
    return 0


def _cmd_codec_decode(args: argparse.Namespace) -> int:
    bundle = np.load(args.input)
    codebooks = list(bundle["codebooks"])
    frontend = CodecFrontend(
        n_codebooks=len(codebooks),
        codebook_size=codebooks[0].shape[0],
        sample_rate=int(bundle["sample_rate"]),
    )
    frontend._codebooks = codebooks  # noqa: SLF001
    waveform = frontend.synthesize(bundle["tokens"], n_iter=args.iterations)
    save_audio(args.output, waveform, frontend.sample_rate)
    print(f"wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="roomkit",
        description="Speech enhancement and room-acoustics toolkit.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    denoise_parser = subparsers.add_parser(
        "denoise", help="remove stationary noise from a recording"
    )
    denoise_parser.add_argument("input", help="noisy input wav file")
    denoise_parser.add_argument("output", help="enhanced output wav file")
    denoise_parser.add_argument(
        "--method", choices=("wiener", "spectral"), default="wiener"
    )
    denoise_parser.add_argument(
        "--noise-frames",
        type=int,
        default=10,
        help="leading frames treated as noise only",
    )
    denoise_parser.set_defaults(func=_cmd_denoise)

    dereverb_parser = subparsers.add_parser(
        "dereverb", help="shorten the reverberant tail with WPE"
    )
    dereverb_parser.add_argument("input", help="reverberant input wav file")
    dereverb_parser.add_argument("output", help="dereverberated output wav file")
    dereverb_parser.add_argument("--taps", type=int, default=8)
    dereverb_parser.add_argument("--delay", type=int, default=3)
    dereverb_parser.add_argument("--iterations", type=int, default=3)
    dereverb_parser.set_defaults(func=_cmd_dereverb)

    vad_parser = subparsers.add_parser(
        "vad", help="detect voice activity and print segments as JSON"
    )
    vad_parser.add_argument("input", help="input wav file")
    vad_parser.add_argument(
        "--output", help="optional file for the JSON segments (default: stdout)"
    )
    vad_parser.add_argument("--threshold-db", type=float, default=6.0)
    vad_parser.set_defaults(func=_cmd_vad)

    diarize_parser = subparsers.add_parser(
        "diarize", help="answer who spoke when; segments as JSON"
    )
    diarize_parser.add_argument("input", help="input wav file")
    diarize_parser.add_argument(
        "--output", help="optional file for the JSON segments (default: stdout)"
    )
    diarize_parser.add_argument(
        "--num-speakers", type=int, default=None, help="number of speakers if known"
    )
    diarize_parser.set_defaults(func=_cmd_diarize)

    rt60_parser = subparsers.add_parser(
        "rt60", help="estimate RT60 from a measured impulse response"
    )
    rt60_parser.add_argument("input", help="impulse response wav file")
    rt60_parser.set_defaults(func=_cmd_rt60)

    codec_parser = subparsers.add_parser("codec", help="discrete token frontend")
    codec_sub = codec_parser.add_subparsers(dest="codec_command")
    encode_parser = codec_sub.add_parser("encode", help="fit codebooks and encode")
    encode_parser.add_argument("input", help="input wav file")
    encode_parser.add_argument("output", help="token bundle (.npz)")
    encode_parser.add_argument("--codebooks", type=int, default=4)
    encode_parser.add_argument("--codebook-size", type=int, default=32)
    encode_parser.set_defaults(func=_cmd_codec_encode)
    decode_parser = codec_sub.add_parser("decode", help="resynthesize from tokens")
    decode_parser.add_argument("input", help="token bundle (.npz)")
    decode_parser.add_argument("output", help="output wav file")
    decode_parser.add_argument("--iterations", type=int, default=32)
    decode_parser.set_defaults(func=_cmd_codec_decode)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    return func(args)
