import polars as pl


def test_iteration(prices_pl: pl.DataFrame):
    # iterate
    for row in prices_pl.rows(named=True):
        row


def test_index(prices_pl: pl.DataFrame):
    index = prices_pl["date"].to_list()
    print(index)


def test_index_history(prices_pl: pl.DataFrame):
    index = prices_pl["date"].to_list()

    for i in range(len(index)):
        up_to_now = index[: i + 1]
        print(up_to_now)


def test_index_window(prices_pl: pl.DataFrame):
    index = prices_pl["date"].to_list()

    window_size = 3

    for i in range(len(index)):
        window = index[max(0, i - window_size + 1) : i + 1]
        print(window)
