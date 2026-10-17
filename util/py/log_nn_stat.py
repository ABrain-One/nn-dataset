import argparse

from ab.nn.util.NNAnalysis import log_nn_stat


def main():
    parser = argparse.ArgumentParser(description="Calculate statistics for LEMUR models.")
    parser.add_argument('--nn', type=str, help="Filter by neural network name")
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of models to process")
    parser.add_argument('--rewrite', action='store_true',
                        help="Recompute statistics for models that already have a stat file "
                             "(needed after adding a new metric such as nn_depth)")
    args = parser.parse_args()

    print(f"Fetching data with filters: nn={args.nn}, limit={args.limit}, rewrite={args.rewrite}")

    try:
        # Fetch data using the API
        log_nn_stat(args.nn, max_rows=args.limit, rewrite=args.rewrite)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

if __name__ == "__main__":
    main()