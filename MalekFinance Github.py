import pandas as pd
from scipy.stats import skew
import MalekFinance as MF
import pyfolio as pf
from matplotlib import pyplot as plt
plt.style.use('ggplot')

def drawdowns(x,i=5):
    """
    --------------------------------------------
    i=5 as Default
    
    i = The number of Drawdowns returned

    --------------------------------------------
    
    """
    drawdownVW = pd.DataFrame()
    drawdownVW['cumulative_returns'] = (1 + x).cumprod()
    drawdownVW['rolling_max'] = drawdownVW['cumulative_returns'].cummax()
    drawdownVW['drawdown'] = drawdownVW['cumulative_returns'] / drawdownVW['rolling_max'] - 1
    max = []
    start_dates_list = []
    end_dates_list =[]
    for _ in range(i):
        max.append(drawdownVW.drawdown.min())
        dd_peak_date = drawdownVW[drawdownVW.drawdown == drawdownVW.drawdown.min()].index[0]
        start_date = drawdownVW[:dd_peak_date][drawdownVW['cumulative_returns'][:dd_peak_date] == drawdownVW['cumulative_returns'][:dd_peak_date].max()].index[0]
        start_dates_list.append(start_date)
        drawdownVW = drawdownVW[~drawdownVW.index.isin(drawdownVW[start_date:dd_peak_date].index)]
        end_date = 'g'
        try:
            end_date = drawdownVW[start_date:][drawdownVW.drawdown == 0][:1].index[0]
        except:
            end_date = drawdownVW[-1:].index[0]
        end_dates_list.append(end_date)
        drawdownVW = drawdownVW[~drawdownVW.index.isin(drawdownVW[start_date:end_date].index)]
    df = pd.DataFrame(data=max,index=[start_dates_list,end_dates_list])
    df.columns = [f'{i} Worst Drawdowns']
    df = round(df*100,3)
    drawdown_lengths = []
    for start, end in zip(df.index.get_level_values(0), df.index.get_level_values(1)):
        drawdown_lengths.append(len(pd.date_range(start=start, end=end, freq='M')))
    df['Drawdown Length in Months'] = drawdown_lengths
    df['Drawdown Length in Months'] -=1
    return df

def regression_OLS(y,x,i=0):
    """
    --------------------------------------------
    i=0 as Default,

    if i = 0, regression is run non robust SE

    for robust SEs, set i to cov_type of choice

    Example:

    Newey West SEs, i ='HC0'

    mf.regression(y,x,'HC0')

    --------------------------------------------
    """
    import statsmodels.api as sm
    x = sm.add_constant(x)
    if i == 0:
        reg = sm.OLS(y,x).fit()
    else:
        reg = sm.OLS(y,x).fit(cov_type=i)
    print(reg.summary())
    return reg

def regression_clustered_OLS(y,x):
    import statsmodels.api as sm
    x = sm.add_constant(x)    
    reg = sm.OLS(y,x).fit(cov_type='cluster',cov_kwds={'groups':y.index.get_level_values(0)})
    print(reg.summary())
    return reg

def CRSP_US_RET():
    data = pd.read_csv('/Users/malek/Documents/WRDS Data/CRSP Returns Data All Exchanges 63-22.csv',low_memory=False)
    data = data.drop(['PRIMEXCH'],axis=1)
    data = data.drop(['EXCHCD'],axis=1)
    data.date = pd.to_datetime(data.date,dayfirst=True)
    data = data[~data.duplicated(subset=['PERMNO', 'date'], keep='first')]
    data1 = data.pivot(index='date', columns='PERMNO', values='RET')
    data1 = data1.apply(pd.to_numeric, errors='coerce')
    return data1

def CRSP_US_SIZE():
    data_size = pd.read_csv('/Users/malek/Documents/WRDS Data/CRSP Size Data All Exchanges 63-22.csv',low_memory=False)
    data_size.date = pd.to_datetime(data_size.date,dayfirst=True)
    data_size['Market Cap']= (data_size['PRC']/data_size['CFACPR'])*(data_size['SHROUT']*data_size['CFACSHR'])
    data_size = data_size[~data_size.duplicated(subset=['PERMNO', 'date'], keep='first')]
    data_size1 = data_size.pivot(index='date', columns='PERMNO', values='Market Cap')
    data_size1 = data_size1.abs()
    data_size1 = data_size1.apply(pd.to_numeric,errors='coerce')
    monthly_size = data_size1.resample('M').last()
    monthly_size = monthly_size*1000
    return monthly_size
    
def sharpe(x=pd.DataFrame,RF=0,i=0):
    """ 
    --------------------------------------------
    Parameters = (x,RF=0,i=0)
    --------------------------------------------
    --------------------------------------------
    x = The Time Series

    --------------------------------------------
    RF=0 as Default,

    if RF = 0: Sharpe Ratio Formula = μ/σ2       
    if RF = 1: Sharpe Ratio Formula = (μ-Rf)/σ2
    --------------------------------------------
    i=0 as Default,

    i=Column of Dataframe
    x.iloc[:,i]
    --------------------------------------------
    """
    if RF == 1:
        RF = MF.RiskFree()
        RF = RF[RF.index.isin(x.index)]
        x = x[x.index.isin(RF.index)]
        x = pd.DataFrame(x.iloc[:,0] - RF.iloc[:,0])
        A = x.iloc[:,i][:].mean()*12
        S = x.iloc[:,i][:].std()*(12**0.5)
        return print(f'Sharpe Ratio rf: {"{:.2f}".format(round(A/S,2))}')
    else:
        A = x.iloc[:,i][:].mean()*12
        S = x.iloc[:,i][:].std()*(12**0.5)
        return print(f'Sharpe Ratio: {"{:.2f}".format(round(A/S,2))}')

def MDD(x):
    drawdownVW = pd.DataFrame()
    drawdownVW['cumulative_returns'] = (1 + x).cumprod()
    drawdownVW['rolling_max'] = drawdownVW['cumulative_returns'].cummax()
    drawdownVW['drawdown'] = drawdownVW['cumulative_returns'] / drawdownVW['rolling_max'] - 1
    return print(f'Max Drawdown: {"{:.1f}".format(round(drawdownVW.drawdown.min()*100,1))}%')

def RiskFree(x=0):
    """
    --------------------------------------------
    Parameters = (x)
    --------------------------------------------
    x=0 as Default,

    if x=0: RiskFree Rate from 1926-2023
    if x=1: RiskFree Rate from 1992-2022
    --------------------------------------------
    """

    RiskFree = pd.read_csv('/Users/malek/Documents/WRDS Data/US Risk Free Rate Ken French.csv',index_col=0)
    RiskFree.index = pd.to_datetime(RiskFree.index,format='%Y%m')
    RiskFree = RiskFree.resample('M').last()
    if x == 0:
        return RiskFree
    elif x == 1:
        return RiskFree[786:-11]
    else:
        raise ValueError('Error \n\n                   x should = 0 or 1 \n\n                   if x=0 RiskFree Rate from 1926-2023 \n                   if x=1: RiskFree Rate from 1992-2022')
    
def mean(x,i=0):
    return print(f'Annual Return: {"{:.2f}".format(round(x.iloc[:,i][:].mean()*1200,2))}%')

def vol(x,i=0):
    return print(f'Annual Volity: {"{:.2f}".format(round(x.iloc[:,i][:].std()*(12**0.5)*100,2))}%')
    
def summary(x,i=0):
    """
    --------------------------------------------
    i=0 as Default,

    if i = 0: Sharpe Ratio Formula = μ/σ2       
    if i = 1: Sharpe Ratio Formula = (μ-Rf)/σ2
    --------------------------------------------
    """
    return (MF.mean(x),MF.vol(x),MF.sharpe(x,i),MF.MDD(x))[0]





