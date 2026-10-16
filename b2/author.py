"""Maintenance only: python3 -m b2.author SOURCE_JSONL UPSTREAM_LICENSE OUTPUT_DIR."""
import argparse
from .dataset import author


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("license")
    parser.add_argument("output")
    args = parser.parse_args()
    result = author(args.source, args.license, args.output)
    print(f"Validated 42 worlds; bundle SHA-256 {result['bundle_sha256']}")


if __name__ == "__main__":
    main()
