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
        dest="dry",
        metavar="DRY",
    )

    args = parser.parse_args()
    source = args.source
    out_folder = args.dest

    os.makedirs(out_folder, exist_ok=True)

    separate(source, out_folder, device=args.device, dry=args.dry)


def separate(
    in_folder: str,
    out_folder: str,
    model: str = "mdx_extra",
    device: str = "cuda",
    dry: bool = False,
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

    if dry is not None:
        print("Found audio files:")
        for f in files:
            print(f)

        print("\nDRY RUN, not separating audio")
        return

    print("Separating:")
    print("\n".join(files))

    run_cmd(cmd + files)

    print("fixing folder structure")
    for root, _, fs in os.walk(out_folder):

        if "other.wav" not in fs:
            continue

        for f in fs:
            stem = f.split(".")[0].upper()
            os.makedirs(os.path.join(root, stem), exist_ok=True)
            cmd = ["mv", "-u", os.path.join(root, f), os.path.join(root, stem)]
            run_cmd(cmd)


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

        run_cmd(cmd)


    print("DONE")


if __name__ == "__main__":
    main()
