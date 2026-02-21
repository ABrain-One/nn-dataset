"""
Check training results and display MAE in human-readable format (years).

The framework stores a normalized accuracy value (higher = better):
    normalized = 1 - (MAE_years / 20.0)
So to recover actual MAE:
    MAE_years = (1 - normalized) * 20.0

Usage:
    python check_results.py
"""
import ab.nn.api as api

MAE_THRESHOLD = 20.0  # must match create_metric() in mae.py


def to_mae(normalized: float) -> float:
    """Convert framework normalized accuracy back to MAE in years."""
    return (1.0 - normalized) * MAE_THRESHOLD


if __name__ == '__main__':
    print("=" * 60)
    print("AGE ESTIMATION RESULTS  (UTKFace / MobileAgeNet)")
    print("=" * 60)

    df = api.data(task='age-regression', only_best_accuracy=True)

    if df.empty:
        print("\nNo results found. Run training first.")
        exit(0)

    # Add MAE column in years
    df = df.copy()
    df['mae_years'] = df['accuracy'].apply(lambda a: to_mae(float(a)))

    print(f"\n{'Model':<30} {'Dataset':<12} {'MAE (years)':>12} {'Normalized':>12}")
    print("-" * 70)

    for _, row in df.iterrows():
        nn_name   = row.get('nn', '?')
        dataset   = row.get('dataset', '?')
        acc       = float(row['accuracy'])
        mae_years = to_mae(acc)
        print(f"{nn_name:<30} {dataset:<12} {mae_years:>11.2f}y {acc:>12.4f}")

    print("-" * 70)

    best = df.loc[df['mae_years'].idxmin()]
    print(f"\nBEST RESULT")
    print(f"  Model      : {best['nn']}")
    print(f"  Dataset    : {best['dataset']}")
    print(f"  MAE        : {best['mae_years']:.2f} years")
    print(f"  Normalized : {float(best['accuracy']):.4f}")
    print()
    print(f"  >> For your paper: MAE = {best['mae_years']:.2f} years <<")
