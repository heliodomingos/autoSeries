import matplotlib.pyplot as plt
import seaborn as sns


def plot_series(series, target_value, loc, tittle=""):
    plt.figure(figsize=(8, 8))
    plt.title(tittle)
    plt.plot(series.ix[:, 0])
    plt.xlabel("Time")
    plt.ylabel(target_value)
    plt.savefig(loc)

def plot_series_two_line(series1, series2, target_value, loc, tittle=""):
    plt.figure(figsize=(8, 8))
    plt.title(tittle)
    plt.plot(series1.ix[:, 0], label="Observed Values")
    plt.plot(series2.ix[:, 1], label="Predicted Values")
    plt.legend(loc="best")
    plt.xlabel("Time")
    plt.ylabel(target_value)
    plt.savefig(loc)


def plot_diff(data, key, type_f):
    d = data[key][type_f]['Predictions'][['Diff']].copy()
    d.index.name = 'Date'
    d.reset_index(level=0, inplace=True)
    d['Date'].values.astype(float)
    d['timestamps'] = d.index
    plt.cla()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns_plot = sns.regplot(ax=ax, x='timestamps', y='Diff', data=d, fit_reg=True, order=1, label='linear')
    sns_plot = sns.regplot(ax=ax, x='timestamps', y='Diff', data=d, fit_reg=True, order=2, label='quadratic')
    sns_plot = sns.regplot(ax=ax, x='timestamps', y='Diff', data=d, fit_reg=True, order=3, label='polynomial(3)')
    plt.legend(loc="best")
    plt.axis('tight')
    plt.grid()
    fig = sns_plot.get_figure()
    fig.savefig("report/" + key + "/diff_reg_" + type_f + ".png", fig_size=(4, 8), bbox_inches='tight')
    plt.cla()
    data[key][type_f]['diff_plot'] = "report/" + key + "/diff_reg_" + type_f + ".png"
