import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


REGIONS = ['SGP', 'HKG']
FEATURES = ['rgdpe', 'emp', 'hc', 'labsh', 'cn']
FEATURES_EXTRA = ['ctfp', 'ccon', 'cda']


def split(dataframe, key, labels):
    return [dataframe.loc[dataframe[key] == label].reset_index(drop=True) for label in labels]


def read(filename, labels):
    dataframe_read = pd.read_csv(filename)
    regions = split(dataframe_read, 'RegionCode', REGIONS)
    ret = {}
    for region, region_dataframe in zip(REGIONS, regions):
        features = split(region_dataframe, 'VariableCode', labels)
        column_years = features[0]['YearCode']
        columns = []
        columns.append(column_years)
        for feature, feature_dataframe in zip(labels, features):
            columns.append(feature_dataframe.rename(columns={'AggValue': feature})[feature])
        ret[region] = pd.concat(columns, axis=1)
    return ret, column_years
        

def save(filename):
    plt.savefig(filename, bbox_inches='tight')


def plot_time(data, features, title, ylabel, scale=1, ymin=None):
    plt.clf()
    for name, dataframe in data.items():
        plt.plot(dataframe['YearCode'], scale * dataframe[features], label=name)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if ymin:
        plt.ylim(ymin=ymin)


def plot_3d(data, xlabel, ylabel, zlabel, title):
    plt.clf()
    ax = plt.axes(projection='3d')
    for name, dataframe in data.items():
        ax.scatter3D(dataframe[xlabel], dataframe[ylabel], dataframe[zlabel], label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.legend()


def make_tables(data):
    for name, dataframe in data.items():
        print(name)
        print(dataframe.to_latex( index=False, float_format="{:0.2f}".format ))
        print()


def linear_regression(data, xlabels, ylabel):
    ret = {}
    for name, dataframe in data.items():
        X, y = dataframe[xlabels].values[1:], dataframe[ylabel].values[1:] # first row is NaN
        fit = LinearRegression().fit(X, y)
        ret[name] = (fit.coef_, fit.intercept_, fit.score(X, y))
    return ret


def pct_change(data):
    data_pct_change = {name: dataframe.pct_change() * 100 for name, dataframe in data.items()}
    for _, dataframe in data_pct_change.items():
        dataframe['YearCode'] = column_years
    return data_pct_change


data, column_years = read('data/sg_hk_data.csv', FEATURES)
data_pct_change = pct_change(data)

'''
plot_time(data, ['rgdpe'], 'Real GDP', '2017 USD (millions)')
save('plots/rgdpe_time.png')
'''

'''
plot_3d(data_pct_change, 'cn', 'emp', 'rgdpe', 'Real GDP Growth vs Capital and Labor Growth') 
save('plots/rgdpeg_cng_empg.png')
'''

#fit = linear_regression(data_pct_change, ['emp', 'cn'], 'rgdpe')
#print(fit)
#fit = linear_regression(data_pct_change, ['YearCode'], 'rgdpe')
#print(fit)


def decompose(data, ref_data):
    ret = {}
    for (name, dataframe), (_, ref_dataframe) in zip(data.items(), ref_data.items()):
        al = ref_dataframe['labsh']
        ak = 1 - al
        cn = ak * dataframe['cn']
        emp = al * dataframe['emp']
        tfp = dataframe['rgdpe'] - cn - emp
        ret[name] = pd.concat([dataframe['YearCode'], cn, emp, tfp], axis=1, keys=['YearCode', 'cn', 'emp', 'tfp'])
    return ret


'''
contributions = decompose(data_pct_change, data)
plot_time(data_pct_change, ['rgdpe'], 'Real GDP Growth', '% Change')
save('plots/rgdpe_growth.png')
plot_time(contributions, ['cn'], 'Real GDP Growth Capital Contribution', '% Change')
save('plots/rgdpe_growth_captial_contribution.png')
plot_time(contributions, ['emp'], 'Real GDP Growth Labor Contribution', '% Change')
save('plots/rgdpe_growth_labor_contribution.png')
plot_time(contributions, ['tfp'], 'Real GDP Growth TFP Contribution', '% Change', ymin=-11)
save('plots/rgdpe_growth_tfp_contribution.png')
'''


def decade_summary(data, group_size=10):
    ret = {}
    for name, dataframe in data.items(): 
        ret[name] = dataframe.groupby(np.arange(len(dataframe)) // group_size).mean()
        ret[name]['YearCode'] = (ret[name]['YearCode'] // 10 * 10).astype(int)
    return ret

'''
make_tables(decade_summary(data))
make_tables(decade_summary(data_pct_change))
'''

def decompose2(data, ref_data):
    ret = {}
    for (name, dataframe), (_, ref_dataframe) in zip(data.items(), ref_data.items()):
        print(dataframe)
        al = ref_dataframe['labsh']
        ak = 1 - al
        cn = ak * dataframe['cn']
        emp = al * dataframe['emp']
        hc = al * dataframe['hc']
        tfp = dataframe['rgdpe'] - cn - emp - hc
        ret[name] = pd.concat([dataframe['YearCode'], cn, emp, hc, tfp], axis=1, keys=['YearCode', 'cn', 'emp', 'hc', 'tfp'])
    return ret


contributions2 = decompose2(data_pct_change, data)

'''
plot_time(data, ['hc'], 'Human Capital', 'Ratio to Reference Country')
save(f'plots/hc/hc_time.png')
plot_time(data_pct_change, ['hc'], 'Human Capital Growth', '% change')
save(f'plots/hc/hc_growth.png')
plot_time(contributions2, ['cn'], 'Real GDP Growth Capital Contribution', '% Change')
save(f'plots/hc/rgdpe_growth_captial_contribution.png')
plot_time(contributions2, ['emp'], 'Real GDP Growth Labor Contribution', '% Change')
save(f'plots/hc/rgdpe_growth_labor_contribution.png')
plot_time(contributions2, ['hc'], 'Real GDP Growth Human Capital Contribution', '% Change')
save(f'plots/hc/rgdpe_growth_hc_contribution.png')
plot_time(contributions2, ['tfp'], 'Real GDP Growth TFP Contribution', '% Change', ymin=-11)
save(f'plots/hc/rgdpe_growth_tfp_contribution.png')
'''

plot_time(data, ['cn'], 'Capital', '2017 USD (millions)')
save(f'plots/cn_time.png')
plot_time(data, ['emp'], 'Labor', 'Population in millions')
save(f'plots/emp_time.png')


data2, column_years = read('data/sg_hk_data_extra.csv', FEATURES_EXTRA)
data_pct_change2 = pct_change(data2)


def decompose3(data, ref_data):
    ret = {}
    for (name, dataframe), (_, ref_dataframe) in zip(data.items(), ref_data.items()):
        ccon = ref_dataframe['ccon']
        cda = ref_dataframe['cda']
        inv = cda - ccon
        ret[name] = pd.concat([dataframe['YearCode'], ccon, cda, inv], axis=1, keys=['YearCode', 'ccon', 'cda', 'inv'])
    return ret


contributions3 = decompose3(data_pct_change2, data2)

# read fdi file
with open('data/sg_hk_fdi.csv', 'r') as fin:
    lines = fin.readlines()
    lines = [list(map(lambda x: x.strip('\n').strip('"'), line.split(','))) for line in lines]
    lines = [line[6:-3] for line in lines if line[0] == 'Singapore' or line[0] == 'Hong Kong SAR']
    lines = [[float(v) / 10e6 if v != '' else 0 for v in line] for line in lines]

contributions3['HKG']['fdi'] = lines[0]
contributions3['SGP']['fdi'] = lines[1]

plot_time(data_pct_change2, ['ctfp'], 'PWT TFP Growth', 'TFP')
save(f'plots/pwt_tfp_delta_time.png')
plot_time(data2, ['ctfp'], 'PWT TFP', 'TFP')
save(f'plots/pwt_tfp_time.png')
plot_time(data2, ['ccon'], 'Consumption', '2017 USD (millions)')
save(f'plots/ccon_time.png')
plot_time(contributions3, ['inv'], 'Investment', '2017 USD (millions)')
save(f'plots/inv_time.png')
plot_time(contributions3, ['fdi'], 'Foreign Investment', '2017 USD (millions)')
save(f'plots/fdi_time.png')


def decompose4(data, ref_data):
    ret = {}
    for (name, dataframe), (_, ref_dataframe) in zip(data.items(), ref_data.items()):
        ccon = ref_dataframe['ccon']
        cda = ref_dataframe['cda']
        inv = cda - ccon
        cn = dataframe['cn']
        delta = 1 - (cn - inv.shift(1)) / cn.shift(1)
        ret[name] = pd.concat([dataframe['YearCode'], delta], axis=1, keys=['YearCode', 'delta'])
    return ret

contributions4 = decompose4(data, data2)


plot_time(contributions4, ['delta'], 'Deprecation', 'Ratio')
save(f'plots/delta_time.png')


def decompose5(data, ref_data):
    ret = {}
    for (name, dataframe), (_, ref_dataframe) in zip(data.items(), ref_data.items()):
        rgdpeL = dataframe['rgdpe'] / dataframe['emp']
        cnL = dataframe['cn'] / dataframe['emp']
        ret[name] = pd.concat([dataframe['YearCode'], rgdpeL, cnL], axis=1, keys=['YearCode', 'rgdpeL', 'cnL'])
    return ret


contributions5 = decompose5(data, data)
plot_time(contributions5, ['rgdpeL'], 'GDP Per Captia', '2017 USD')
save(f'plots/rgdpeL_time.png')
plot_time(contributions5, ['cnL'], 'Capital Per Worker', '2017 USD')
save(f'plots/cnL_time.png')

data_pct_change3 = pct_change(contributions5)
plot_time(data_pct_change3, ['rgdpeL'], 'GDP Per Capita Growth', '% Change')
save(f'plots/rgdpeL_growth.png')
plot_time(data_pct_change3, ['cnL'], 'Capital Per Worker Growth', '% Change')
save(f'plots/cnL_growth.png')