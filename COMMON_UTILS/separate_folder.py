import os
import sys
import argparse
import subprocess as sp

from utils import get_audio_files, run_cmd


def main():
    parser = argparse.ArgumentParser(
        description="Separate a folder of audio into their individual stems",
    )

    parser.add_argument(
        "source",
        help="Source directory of audio",
    )

    parser.add_argument(
        "dest",
        help="Folder to save processed track data",
    )

    parser.add_argument(
        "-d",
        "--device",
        help="Device to run Demucs on. Defualt cuda",
        choices=["cpu", "cuda"],
        default="cuda",
        required=False,
    )

    parser.add_argument(
        "--dry",
        help="Perform dry run (list files that it would separate otherwise)",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--force",
        help="Separate all tracks, even if found in DEST",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    source = args.source
    out_folder = args.dest

    os.makedirs(out_folder, exist_ok=True)

    separate(source, out_folder, device=args.device, dry=args.dry, force=args.force)


def separate(
    in_folder: str,
    out_folder: str,
    model: str = "mdx_extra",
    device: str = "cuda",
    dry: bool = False,
    force: bool = False,
):
    cmd = [
        "python3",
        "-m",
        "demucs.separate",
        "-o",
        str(out_folder),
        "-n",
        model,
        "-d",
        device,
    ]

    files = get_audio_files(in_folder)

    if len(files) == 0:
        print(f"No files found in {os.path.abspath(in_folder)}")
        return

    if not force:
        new_files = []
        done_files = set()

        for df in os.listdir(out_folder):
            done_files.add(df)

        for f in files:
            fn = os.path.split(f)[-1]
            if fn[:-4] not in done_files:
                new_files.append(f)

        files = new_files

    if dry:
        print("Found audio files:")
        for f in files:
            print(f)

        print("\nDRY RUN, not separating audio")
        return



    print("Separating:")
    print("\n".join(files))

    if len(files) > 0:
        status = run_cmd(cmd + files)
        if status != 0:
            print(status)
            return

    print("fixing folder structure")
    for root, dirs, fs in os.walk(out_folder):

        if "other.wav" in fs:
            if len(fs) != 4:
                continue
        else:
            continue

        for f in fs:
            stem = f.split(".")[0].upper()
            os.makedirs(os.path.join(root, stem), exist_ok=True)
            cmd = ["mv", "-u", os.path.join(root, f), os.path.join(root, stem)]
            status = run_cmd(cmd)


    print("copying source tracks to processed folder")
    for f in files:
        name = f.split("/")[-1]
        ext = name.split(".")[-1]
        name = ".".join(name.split(".")[:-1])
        cmd = [
            "cp",
            str(f),
            str(os.path.join(out_folder, name, "source." + ext)),
        ]

        status = run_cmd(cmd)

    print("DONE")


if __name__ == "__main__":
    main()
