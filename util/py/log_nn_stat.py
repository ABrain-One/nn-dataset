import argparse

from ab.nn.util.NNAnalysis import log_nn_stat


def main():
    parser = argparse.ArgumentParser(description="Calculate statistics for LEMUR models.")
    parser.add_argument('--nn', type=str, help="Filter by neural network name")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of models to process")
    args = parser.parse_args()

    print(f"Fetching data with filters: nn={args.nn}, limit={args.limit}")

    try:
        # Fetch data using the API
        log_nn_stat(args.nn, max_rows=args.limit)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

if __name__ == "__main__":
    main()