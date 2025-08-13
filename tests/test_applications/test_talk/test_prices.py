"""Test prices."""


def test_prices(prices):
    """Check if all prices are positive."""
    for name, data in prices.items():
        x = data.dropna()
        assert (x > 0).all(), f"Negative price for {name} detected"
