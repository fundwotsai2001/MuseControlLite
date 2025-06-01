import os
from pydub import AudioSegment

# ─── CONFIGURE THESE FOUR VARIABLES ─────────────────────────────────────────

# 1) Full path to the root folder that contains subfolders of 47 s audio files:
INPUT_ROOT = r"/home/fundwotsai/Music-Controlnet-light/MuseControlLite_v2/camera_ready_6_1_outpainting"

# 2) Full path to where you want to save the trimmed outputs:
OUTPUT_ROOT = r"/home/fundwotsai/Music-Controlnet-light/MuseControlLite_v2/camera_ready_6_1_outpainting_trimmed"

# 3) Start time (in seconds) of the segment you want to extract:
START_SEC = 24

# 4) End time (in seconds) of the segment you want to extract:
END_SEC = 32


# ─── (No need to change below this line) ────────────────────────────────────

def trim_and_save(input_path: str, output_path: str, start_ms: int, end_ms: int):
    """
    Load an audio file from input_path, trim it to [start_ms:end_ms] (in milliseconds),
    and export it to output_path using the same file format.
    """
    # Attempt to load the audio
    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)  # total duration in milliseconds

    # If the file is shorter than the requested end_ms, skip or adjust.
    # Here we simply skip files shorter than END_SEC seconds.
    if duration_ms < end_ms:
        print(f"  ↳ Skipping (too short): {input_path}")
        return

    # Trim to the requested segment
    segment = audio[start_ms:end_ms]

    # Determine format from original file extension
    ext = os.path.splitext(input_path)[1].lower().lstrip(".")
    if ext == "":
        print(f"  ↳ Cannot determine format for: {input_path}, skipping.")
        return

    # Make sure parent directory for output exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Export the trimmed segment
    segment.export(output_path, format=ext)
    print(f"  ↳ Saved trimmed audio to: {output_path}")


def main():
    # Convert start/end times to milliseconds
    start_ms = START_SEC * 1000
    end_ms = END_SEC * 1000

    # Walk through every subdirectory of INPUT_ROOT
    for root, _, files in os.walk(INPUT_ROOT):
        for fname in files:
            # Only process common audio extensions—add more if needed
            if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a")):
                continue

            full_in_path = os.path.join(root, fname)

            # Compute the relative path of this file w.r.t. INPUT_ROOT
            rel_dir = os.path.relpath(root, INPUT_ROOT)  # e.g. "subfolder1/subfolder2"
            # Build the corresponding output directory path
            out_dir = os.path.join(OUTPUT_ROOT, rel_dir)
            # Output filename is identical to input filename
            full_out_path = os.path.join(out_dir, fname)

            try:
                trim_and_save(full_in_path, full_out_path, start_ms, end_ms)
            except Exception as e:
                print(f"  ↳ Error processing {full_in_path}: {e}")


if __name__ == "__main__":
    main()
