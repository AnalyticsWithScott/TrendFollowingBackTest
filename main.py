# Import standard libraries
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Filter warnings
warnings.filterwarnings('ignore')

# Third-party imports
import yfinance as yf

# Local application imports
from config import getStratStats
from donchian_class import DonchianStrategy
from movingAverage_class import MovingAvgStrategy



# Inputs
stock_tickers = [
    'ANF', 'AMD', 'BITO', 'CANE', 'DAX', 'DDOG', 'DECK', 'DKNG',
    'GLD', 'META', 'NFLX', 'NVDA', 'PINS', 'RBLX', 'SHOP', 'TSLA']
backtest_start_date = '2022-01-01'
backtest_end_date = '2024-01-31'
starting_account_balance = 10000
atr_lookback = 25
atr_stop_multiplier = 2
donchian_entry_signal_lookback = 30
donchian_exit_signal_lookback = 50 
fast_ma = 50
slow_ma = 200
max_risk_per_trade = 0.01
max_position_size = 0.15
no_leverage_position_sizing = 0.05
shorts = False
no_vol_target = False
vol_target = True
no_leverage=False
leverage=True



# Implementing Donchian Strategy Class NO LEVERAGE
donchian_sys_no_leverage = DonchianStrategy(
    tickers=stock_tickers, 
    init_account_size=starting_account_balance,
    atr_multiplier=atr_stop_multiplier, 
    r_max=max_risk_per_trade, 
    max_position_size=no_leverage_position_sizing, 
    entry=donchian_entry_signal_lookback,
    exit_=donchian_exit_signal_lookback, 
    atr_periods=atr_lookback, 
    start=backtest_start_date,
    end=backtest_end_date,
    shorts=shorts, 
    vol_target=no_vol_target, 
    leverage=no_leverage
)
donchian_sys_no_leverage.run()
donchian_port_values_no_leverage = donchian_sys_no_leverage.get_portfolio_values()

# Donchian Strategy Class Returns NO LEVERAGE
donchian_returns_no_leverage = donchian_port_values_no_leverage / donchian_port_values_no_leverage.shift(1)
donchian_log_returns_no_leverage = np.log(donchian_returns_no_leverage)
donchian_cum_rets_no_leverage = donchian_log_returns_no_leverage.cumsum()



# Implementing Donchian Strategy Class with Leverage
donchian_sys = DonchianStrategy(
    tickers=stock_tickers, 
    init_account_size=starting_account_balance,
    atr_multiplier=atr_stop_multiplier, 
    r_max=max_risk_per_trade,
    max_position_size=max_position_size, 
    entry=donchian_entry_signal_lookback,
    exit_=donchian_exit_signal_lookback, 
    atr_periods=atr_lookback, 
    start=backtest_start_date,
    end=backtest_end_date,
    shorts=shorts, 
    vol_target=no_vol_target, 
    leverage=leverage
)
donchian_sys.run()
donchian_port_values = donchian_sys.get_portfolio_values()

# Donchian Strategy Class Returns with Leverage
donchian_returns = donchian_port_values / donchian_port_values.shift(1)
donchian_log_returns = np.log(donchian_returns)
donchian_cum_rets = donchian_log_returns.cumsum()



# Implementing Donchian Strategy Class VOLATILITY TARGET
donchian_sys_vol_target = DonchianStrategy(
    tickers=stock_tickers, 
    init_account_size=starting_account_balance,
    atr_multiplier=atr_stop_multiplier, 
    r_max=max_risk_per_trade,
    max_position_size=max_position_size, 
    entry=donchian_entry_signal_lookback,
    exit_=donchian_exit_signal_lookback, 
    atr_periods=atr_lookback, 
    start=backtest_start_date,
    end=backtest_end_date,
    shorts=shorts, 
    vol_target=vol_target, 
    leverage=leverage
)
donchian_sys_vol_target.run()
donchian_port_values_vol_target = donchian_sys_vol_target.get_portfolio_values()

# Donchian Strategy Class Returns VOLATILITY TARGET
donchian_returns_vol_target = donchian_port_values_vol_target / donchian_port_values_vol_target.shift(1)
donchian_log_returns_vol_target = np.log(donchian_returns_vol_target)
donchian_cum_rets_volTarget = donchian_log_returns_vol_target.cumsum()



# Implementing Moving Average Strategy Class with Leverage
mov_avg_sys = MovingAvgStrategy(
    tickers=stock_tickers, 
    init_account_size=starting_account_balance,
    atr_multiplier=atr_stop_multiplier, 
    r_max=max_risk_per_trade,
    max_position_size=max_position_size, 
    fast_ma=fast_ma, 
    slow_ma=slow_ma,
    atr_periods=atr_lookback, 
    start=backtest_start_date,
    end=backtest_end_date,
    shorts=shorts,  
    leverage=leverage
)
mov_avg_sys.run()
moving_average_port_values = mov_avg_sys.get_portfolio_values()

# Moving Average Strategy Class Returns with Leverage
mov_avg_returns = moving_average_port_values / moving_average_port_values.shift(1)
mov_avg_log_returns = np.log(mov_avg_returns)
mov_avg_cum_rets = mov_avg_log_returns.cumsum()



# Compare to SPY baseline
sp500 = yf.download(tickers = '^GSPC', start=backtest_start_date, end=backtest_end_date)
sp500['returns'] = sp500['Close'] / sp500['Close'].shift(1)
sp500['log_returns'] = np.log(sp500['returns'])
sp500['cum_rets'] = sp500['log_returns'].cumsum()
final_balance = starting_account_balance * np.exp(sp500['cum_rets'].iloc[-1])



# Concatenating all statistics
strategies = [
    ('Donchian Strategy', donchian_log_returns),
    ('Donchian Strategy (Vol Target)', donchian_log_returns_vol_target),
    ('Donchian Strategy (No Leverage)', donchian_log_returns_no_leverage),
    ('Moving Average Strategy', mov_avg_log_returns),
    ('Benchmark - S&P500', sp500['log_returns'])
]

dfs = [pd.DataFrame(getStratStats(log_returns), index=[label]) for label, log_returns in strategies]
df_stats = pd.concat(dfs)

print(df_stats)



# Print Additional Information
donchain_ending_bal = round(donchian_port_values[-1], 2)
donchain_ending_bal_volTarget = round(donchian_port_values_vol_target[-1], 2)
donchain_ending_bal_noLeverage = round(donchian_port_values_no_leverage[-1], 2)
mov_avg_ending_bal = round(moving_average_port_values[-1], 2)

formatted_donchain = f"${donchain_ending_bal:,.2f}"
formatted_donchain_volTarget = f"${donchain_ending_bal_volTarget:,.2f}"
formatted_donchain_noLeverage = f"${donchain_ending_bal_noLeverage:,.2f}"
formatted_mov_avg = f"${mov_avg_ending_bal:,.2f}"

print(f'Donchian Strategy Ending Balance = {formatted_donchain}')
print(f'Donchian Strategy Ending Balance (Vol Target) = {formatted_donchain_volTarget}')
print(f'Donchian Strategy Ending Balance (No Leverage) = {formatted_donchain_noLeverage}')
print(f'Moving Average Strategy Ending Balance = {formatted_mov_avg}')



# Graphing returns
donchian_graph_returns = (np.exp(donchian_cum_rets) -1 )* 100
donchian_graph_returns_volTarget = (np.exp(donchian_cum_rets_volTarget) -1 )* 100
donchian_graph_returns_noLeverage = (np.exp(donchian_cum_rets_no_leverage) -1 )* 100
mov_avg_graph_returns = (np.exp(mov_avg_cum_rets) -1 )* 100
spy_graph_returns = (np.exp(sp500['cum_rets']) - 1) * 100

# Plot the strategies to compare performance
plt.figure(figsize=(12, 8))
plt.plot(donchian_graph_returns, label='Donchian Strategy', color = 'Navy')
plt.plot(mov_avg_graph_returns, label='Moving Average Strategy', color = 'Green')
plt.plot(donchian_graph_returns_volTarget, label='Donchian Strategy (Vol Target)', color = 'Red')
plt.plot(donchian_graph_returns_noLeverage, label='Donchian Strategy (No Leverage)', color = 'Purple')
plt.plot(spy_graph_returns, label='S&P 500', color = 'Orange')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Returns (%)')
plt.title('Cumulative Portfolio Returns')
plt.tight_layout()
plt.show()
