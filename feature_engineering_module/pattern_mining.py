from prefixspan import PrefixSpan
from feature_engineering_module.aggregation_operators import derivative


def t(d):
    r = []
    for i in range(len(d)-7):
        r.append(d[i:i+7])
    return r


def find_matches(time_series):
    if 'derivative' not in time_series.keys():
        derivative(time_series)
    describe = time_series['derivative'].describe()
    quartil_25 = describe['25%']
    quartil_50 = describe['50%']
    quartil_75 = describe['75%']

    def f(row):
        if row['derivative'] == 0:
            val = 'a'
        elif row['derivative'] > quartil_75:
            val = 'b'
        elif row['derivative'] > quartil_50:
            val = 'c'
        elif row['derivative'] > quartil_25:
            val = 'd'
        elif row['derivative'] <= quartil_25:
            val = 'e'
        else:
            val = 'f'
        return val
    time_series['prefixspan'] = time_series.apply(f, axis=1)
    print(PrefixSpan(t(time_series['prefixspan'])).topk(100, closed=False))
    return time_series
