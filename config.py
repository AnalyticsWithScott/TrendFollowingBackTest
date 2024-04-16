import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def get_data() -> pd.DataFrame:
    """Fetches data from a local csv file containing information about futures.

    Returns:
        pd.DataFrame: Returns data frame consisting of futures multipliers and other fundamental information.
    """    
    filepath_string = '/Users/scottellsworth/Desktop/Python/Scott_Trend_Following/data/2023-10-06-watchlist.csv'

    futures_data = pd.read_csv(filepath_string)

    return futures_data    

def pull_dictionary(futures_data:pd.DataFrame) -> dict:
    """Grabs the fuures data and manipulates it into a readable format for the YFinance API to access the information.

    Args:
        futures_data (pd.DataFrame): Data frame consisting futures data: symbols, multipliers, tick sizes, etc..

    Returns:
        dict: Returns dictionary of futures symbols in a readable format for the YFinance API.
    """    
    futures_dict = futures_data.set_index('yfinance_tickers').to_dict(orient='index')

    return futures_dict

def getStratStats(log_returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """Calculate various performance statistics for a trading or investment strategy.

    Parameters:
    log_returns (pd.Series): Series of log returns for the strategy.
    risk_free_rate (float): Risk-free rate of return, defaults to 0.05.

    Returns:
    dict: A dictionary containing the following strategy statistics:
        - 'tot_returns': Total returns of the strategy.
        - 'annual_returns': Mean annual returns of the strategy.
        - 'annual_volatility': Annual volatility of the strategy.
        - 'sortino_ratio': Sortino ratio of the strategy.
        - 'sharpe_ratio': Sharpe ratio of the strategy.
        - 'max_drawdown': Maximum drawdown of the strategy.
        - 'max_drawdown_duration': Duration of the maximum drawdown in days.
        - 'risk_adj_return': Risk-adjusted return of the strategy.
    """
    stats = {}  # Total Returns
    stats['tot_returns'] = np.exp(log_returns.sum()) - 1  
  
    # Mean Annual Returns
    stats['annual_returns'] = np.exp(log_returns.mean() * 252) - 1  
  
    # Annual Volatility
    stats['annual_volatility'] = log_returns.std() * np.sqrt(252)
    
    # Sortino Ratio
    annualized_downside = log_returns.loc[log_returns<0].std() * np.sqrt(252)
    stats['sortino_ratio'] = (stats['annual_returns'] - risk_free_rate) / annualized_downside   
    
    # Sharpe Ratio
    stats['sharpe_ratio'] = (stats['annual_returns'] - risk_free_rate) / stats['annual_volatility']
        
    # Max Drawdown
    cum_returns = log_returns.cumsum() - 1
    peak = cum_returns.cummax()
    drawdown = peak - cum_returns
    max_idx = drawdown.argmax()
    stats['max_drawdown'] = 1 - np.exp(cum_returns[max_idx]) / np.exp(peak[max_idx]) 
    
    # Max Drawdown Duration
    strat_dd = drawdown[drawdown==0]
    strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
    strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
    strat_dd_days = np.hstack([strat_dd_days, (drawdown.index[-1] - strat_dd.index[-1]).days])
    stats['max_drawdown_duration'] = strat_dd_days.max()

    stats['risk_adj_return'] = round(stats['annual_returns'] / stats['annual_volatility'],2)

    return {k: np.round(v, 4) if type(v) == np.float_ else v for k, v in stats.items()}