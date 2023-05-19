import pandas as pd

def resample_index(index, rule):
    series = pd.Series(index=index, data=index)
    a = series.resample(rule=rule).first()
    return a.values

    pass

