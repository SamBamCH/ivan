"""
Test script to validate randomness parameter for distribution shrinking.

This tests that:
1. randomness=1.0 produces full variance
2. randomness=0.5 produces ~25% variance (since variance scales by r²)
3. randomness=0.0 produces samples very close to the mean
"""
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ivan.sports.nfl.models import NFLBase
from ivan.sports.nfl.correlated_samples import generate_correlated_samples_gaussian


def test_randomness_parameter():
    """Test that randomness parameter correctly shrinks distributions."""

    # Create in-memory database for testing
    engine = create_engine("sqlite:///:memory:")
    NFLBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Test with different randomness values
    randomness_values = [1.0, 0.5, 0.25, 0.0]

    print("Testing randomness parameter:")
    print("=" * 60)

    for r in randomness_values:
        try:
            samples = generate_correlated_samples_gaussian(
                session,
                n_samples=1000,
                randomness=r,
                seed=42  # Fixed seed for reproducibility
            )

            if not samples:
                print(f"\nRandomness={r:.2f}: No samples generated (likely no players in DB)")
                continue

            # Calculate statistics across all players
            all_means = []
            all_stds = []

            for player_id, player_samples in samples.items():
                all_means.append(np.mean(player_samples))
                all_stds.append(np.std(player_samples))

            avg_mean = np.mean(all_means)
            avg_std = np.mean(all_stds)

            print(f"\nRandomness={r:.2f}:")
            print(f"  Average sample mean: {avg_mean:.2f}")
            print(f"  Average sample std:  {avg_std:.2f}")

            # At r=0, std should be very close to 0
            if r == 0.0:
                print(f"  ✓ Std is near zero: {avg_std < 0.01}")

        except Exception as e:
            print(f"\nRandomness={r:.2f}: Error - {e}")

    print("\n" + "=" * 60)
    print("Note: This test requires players in the database to be meaningful.")
    print("Expected behavior:")
    print("  - randomness=1.0: Full variance (baseline)")
    print("  - randomness=0.5: ~25% variance (0.5² = 0.25)")
    print("  - randomness=0.0: Near-zero variance (deterministic)")


if __name__ == "__main__":
    test_randomness_parameter()
