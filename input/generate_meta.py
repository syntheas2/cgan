#!/usr/bin/env python3
import argparse
from pathlib import Path
from pythelpers.utils.generate_meta import generate_metadata


def main():
    parser = argparse.ArgumentParser(description="Generate metadata JSON from any CSV file")
    parser.add_argument("--csv_file", default="data.csv", help="Path to the input CSV file")
    parser.add_argument("--output", "-o", default="meta.json", help="Path to output JSON file")
    parser.add_argument("--sample", "-s", type=int, help="Number of rows to sample (default: all rows)")
    parser.add_argument("--full", "-f", action="store_true", help="Include full statistics in output")
    
    args = parser.parse_args()

    # Get the base directory (where the script is located)
    base_dir = Path(__file__).parent
    
    # Set file paths relative to the script location
    csv_file_path = base_dir / 'features4ausw4linearsvc_train.csv'
    output_path = base_dir / 'features4ausw4linearsvc_train.json'
    
    # Print paths for debugging
    print(f"Script location: {base_dir}")
    print(f"CSV path: {csv_file_path}")
    print(f"Output path: {output_path}")
    
    # Use either hardcoded paths or args
    # Uncomment the following line to use command line arguments
    # csv_file_path = Path(args.csv_file)
    # output_path = Path(args.output)
    
    generate_metadata(
        csv_file_path=csv_file_path,
        output_json_path=output_path,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()