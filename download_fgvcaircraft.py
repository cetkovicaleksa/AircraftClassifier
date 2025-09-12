import argparse
from contextlib import contextmanager
from pathlib import Path
from collections import Counter
from pathvalidate import sanitize_filename
import shutil
from torchvision.datasets import FGVCAircraft
# import logging



@contextmanager
def existent_or_temp(path = None):

    if path is None:
        import tempfile
        temp = tempfile.TemporaryDirectory()

        try:
            yield Path(temp.name)
        finally:
            temp.cleanup()
    else:
        yield Path(path)


def main():
    parser = argparse.ArgumentParser(description="Script to download FGVCAircraft dataset")

    parser.add_argument(
        "-out", "-o", 
        type=Path, 
        required=False,
        default=Path("./data"), 
        help="Output directory to save the dataset (default: ./data)"
    )

    parser.add_argument(
        "--root", "-r",
        type=Path,
        required=False,
        default=None,
        help="Directory with the downloaded FGVCAircraft dataset or where to download it. If ommited uses temp directory."
    )

    parser.add_argument(
        "--splits", "-s",
        nargs='+',
        choices=["train", "val", "trainval", "test"],
        required=False,
        default=["trainval"],
        help="Split the dataset using FGVCAircraft anotations"
    )

    parser.add_argument(
        "--annotation-level", "-a",
        type=str,
        choices=["variant", "family", "manufacturer"],
        default="variant",
        help="Anotation granularity (default: variant)"
    )

    parser.add_argument(
        "--classes", "-c",
        nargs="+",
        required=False,
        default=[],
        help="Subset of classes to include (default: all)"
    )

    # parser.add_argument(
    #     "--verbose", "-v",
    #     action="store_true",
    #     help="Enable detailed debug logging"
    # )

    # parser.add_argument(
    #     "--quiet", "-q",
    #     action="store_true",
    #     help="Suppress info logs, only show warnings/errors"
    # )

    args = parser.parse_args()

    # logging.basicConfig(
    #     format="%(asctime)s [%(levelname)s] %(message)s",
    #     level=(logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO)
    # )

    with existent_or_temp(args.root) as root_dir:
        classes = set(map(lambda c: c.strip(), args.classes))
        
        for split in args.splits:
            split_dir = args.out / Path(split)
            split_dir.mkdir(parents=True, exist_ok=True)
            # logging.info(f"Loading FGVCAircraft {split} dataset")

            dataset = FGVCAircraft(root_dir, split, args.annotation_level, download=True)

            # logging.info(f"Copying images for selected classes to {split_dir}")

            img_counts = Counter()

            for img_path, label in zip(map(Path, dataset._image_files), dataset._labels):
                clazz = dataset.classes[label]

                if classes and clazz not in classes:
                    continue

                clazz_dir = split_dir / sanitize_filename(clazz)
                clazz_dir.mkdir(exist_ok=True)

                # logging.debug(f"Copying {img_path} to {clazz_dir / img_path.name}")
                shutil.copy2(img_path, clazz_dir / img_path.name)

                img_counts[clazz] += 1

            # logging.info(f"Finished copying {split} dataset\n")
            # logging.debug(
            #     f"  Classes: {len(img_counts)}\n" +
            #     f"  Images: {sum(img_counts.values())}\n" +
            #     '\n'.join(f"    {clazz:<{max(map(len, img_counts))}}: {count}" for clazz, count in sorted(img_counts.items()))
            # )

            # does not use private fields, may be more reliable but is slower and does not keep the same file name
            # for i in range(len(dataset)):
            #     img, label = dataset[i]
            #     clazz = dataset.classes[label]

            #     if classes and clazz not in classes:
            #         continue

            #     clazz_dir = split_dir / sanitize_filename(clazz)
            #     clazz_dir.mkdir(exist_ok=True)

            #     img.save(clazz_dir / f"{i}.{(img.format or 'jpg').lower()}")      


if __name__ == "__main__":
    main()                       