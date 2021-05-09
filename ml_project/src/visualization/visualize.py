import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    REPORTS_PATH = '../../reports/'
    df = pd.read_csv('../../data/raw/heart.csv')
    df.describe().to_csv(REPORTS_PATH + 'description.csv')
    df.groupby('target').mean().to_csv(REPORTS_PATH + 'groupby_mean.csv')
    df.groupby('target').std().to_csv(REPORTS_PATH + 'groupby_std.csv')

    corr = df.corr()
    heatmap = sns.heatmap(corr)
    fig = heatmap.get_figure()

    fig.savefig(REPORTS_PATH + 'figures/heatmap.png')
