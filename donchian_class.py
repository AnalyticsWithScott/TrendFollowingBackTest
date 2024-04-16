# Import standard libraries
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy, copy
from typing import List
from typing import Dict

# Filter warnings
warnings.filterwarnings('ignore')

# Third-party imports
import talib
import yfinance as yf

# Local application imports
from config import get_data, pull_dictionary


class DonchianStrategy:

    def __init__(self, tickers:List[str], init_account_size:int, atr_multiplier:int, r_max:float, 
                 max_position_size:float, entry:int, exit_:int, atr_periods:int, start:str, 
                 end:str, shorts:bool, vol_target:bool, leverage:bool) -> None:
        
        """
        Initializes the DonchianStrategy object.

        Parameters:
            tickers (List[str]): List of security symbols to trade.
            init_account_size (int): Initial trading capital.
            atr_multiplier (int): Multiplier used to determine the stop loss distance by multiplying this value with ATR.
            r_max (float): Maximum percentage of account that a trade can risk.
            max_position_size (float): Maximum position allocation of the portfolio allowed per trade.
            entry (int): Number of breakout days to generate a buy signal.
            exit_ (int): Number of breakout days to generate a sell signal.
            atr_periods (int): Number of days used to calculate SMA of true range.
            start (str): First date for getting data (format: 'yyyy-mm-dd').
            end (str): End date for getting data (format: 'yyyy-mm-dd').
            shorts (bool): Allow short positions if True.
            vol_target (bool): Apply volatility targeting if True.
            leverage (bool): Allow for turning on or off leverage/margin.
            risk_target (float): If volatility targeting, you can you use -3%.
            annual_interest_rate (float) = 13.25% is charged annually if on margin.
            daily_interest_rate (float) = annual interest rate / 252 (approximately 252 trading days in a year)
        """
        
        self.tickers = tickers
        self.init_account_size = init_account_size
        self.cash = init_account_size
        self.portfolio_value = init_account_size
        self.atr_multiplier = atr_multiplier
        self.r_max = r_max
        self.max_position_size = max_position_size
        self.entry = entry
        self.exit_ = exit_
        self.atr_periods = atr_periods
        self.start = start
        self.end = end
        self.shorts = shorts
        self.vol_target = vol_target
        self.leverage = leverage
        self.risk_target = -0.03
        self.annual_interest_rate = 0.1325
        self.daily_interest_rate = self.annual_interest_rate / 252
        self.margin_interest = 0

        self._prep_data()

    def _prep_data(self) -> None:
        """Prepares the data for trading strategy.

        Parameters:
        None

        Returns:
        None
        """        
        self.data = self._get_data()
        self._calc_breakouts()
        self._calc_N()

    def _get_data(self) -> pd.DataFrame:
        """Fetches historical data for multiple tickers from Yahoo Finance API.

        Returns:
            pd.DataFrame: Returns multi-index column data frame: [Open, High, Low, Close].
        """        
        df = yf.download(tickers = self.tickers, start=self.start, end=self.end)
        df.drop(['Adj Close', 'Volume'], inplace=True, axis=1)
        df.ffill(inplace=True)
        return df.swaplevel(axis=1)
    
    def _calc_breakouts(self) -> pd.DataFrame:
        """Creates Donchian Channel indicators for buy/sell signals

        Returns:
            pd.DataFrame: Data frame with added indicator columns for buy/sell signals
        """        
        for t in self.tickers:
            # Loops through all tickers to create indicator columns
            self.data[t, 'execution_price'] = self.data[t]['Open'].shift(-1)
            self.data[t, 'EL'] = self.data[t]['High'].rolling(self.entry).max()
            self.data[t, 'ExL'] = self.data[t]['Low'].rolling(self.exit_).min()
            if self.shorts:
                # If parameter "shorts" is True, then additional incidactors are created
                self.data[t, 'ES'] = self.data[t]['Low'].rolling(self.entry).min()
                self.data[t, 'ExS'] = self.data[t]['High'].rolling(self.exit_).max()
                
    def _calc_N(self) -> pd.DataFrame:
        """Calculates the Average True Range (ATR) as a proxy for volatility and places this into the data frame.

        Returns:
            pd.DataFrame: Places ATR calculation column into a the data frame.
        """        
        for t in self.tickers:
            self.data[t,'N'] = talib.ATR(self.data[t, 'High'], self.data[t, 'Low'], self.data[t, 'Close'], timeperiod=25).shift(1)
    
    def _check_multipliers(self) -> Dict:
        """Checks for symbols for leveraged markets.  If futures data then a multiplier is applied as appropriate. Stocks/ETFs are simply 1.

        Returns:
            Dict: A dictionary assigning each symbol with its respective multiplier.
        """        
        multiplier_dict = {}
        futures_data = get_data()
        futures_dict = pull_dictionary(futures_data)
        for t in self.tickers:
            # Checking to see if any tickers are futures symbols.
            if t.endswith('=F'):
                multiplier_dict[t] = futures_dict[t]['multiplier']
        return multiplier_dict
    
    def _check_cash_balance(self, shares:int, price:float, ticker:str) -> int:
        """Checks the current cash balance to see if we can make the trade. If we can make the trade, we also will limit the total position allocation to a 
            specific percentage of the overall portfolio.

        Args:
            shares (int): Number of shares to purchase.
            price (float): The price to buy or sell.
            ticker (str): The market being traded.

        Returns:
            int: Number of shares to buy/sell.
        """        
        if ticker.endswith('=F'):
            # Checks if symbol is a futures market.
            return shares
        if (shares * price)/self.portfolio_value >= self.max_position_size:
            # If the total allocation of the portfolio exceeds this threshold, we limit the percentage of the portfolio.
            shares = int(np.floor(self.portfolio_value*self.max_position_size)/price)
        return shares
    
    def _calc_portfolio_value(self, portfolio:dict) -> float:
        """Iterates through portfolio dictionary to find the portfolio value and adds this value to the current cash position.

        Args:
            portfolio (dict): A dictionary of current positions/portfolio, cash, value and dates.

        Raises:
            ValueError: If the value pulled from the dictionary or the cash variable is not a number, it raises an error.

        Returns:
            float: Returns portfolio value in dollars. 
        """        
        pv = sum([value['value'] for key, value in portfolio.items() if isinstance(value, dict)])
        pv += self.cash
        if np.isnan(pv):
            raise ValueError(f"PV = {pv}\n{portfolio}")
        return pv
    
    def _get_dollar_risk(self) -> float:
        """Takes the current portfolio level and multiplies it by the maximum percentage risk per trade.

        Returns:
            float: Returns dollar value of allotted risk (portfolio value (10,000) * max risk percent (0.01))
        """        
        dollar_risk = self.portfolio_value * self.r_max
        return dollar_risk
    
    def _size_position(self, data:pd.DataFrame, dollar_risk:float, ticker:str) -> int:
        """This checks the multipliers to ensure the adequate number of shares is bought/sold. After assessing multipliers, we normalize risk by diving our dollar risk value by a multiple of ATR.

        Args:
            data (pd.DataFrame): The large data frame consisting all of markets and indicators.
            dollar_risk (float): The maximum dollar risk we are willing to take per trade.
            ticker (str): The market that will be traded.

        Returns:
            int: Returns the number of shares to ensure risk is maintained.
        """        
        multiplier_dict = self._check_multipliers()
        multiplier = multiplier_dict.get(ticker,1)
        shares = int(np.floor(dollar_risk / (data['N'] * self.atr_multiplier * multiplier)))
        return shares
    
    def _run_system(self, ticker:str, data:pd.DataFrame, position:dict) -> dict:
        """This method runs the entire trading strategy and consists of all the signals.

        Args:
            ticker (str): The market that will be traded.
            data (pd.DataFrame): The large data frame consisting all of markets and indicators.
            position (dict): A dictionary consisting of all positon metrics such as shares, entry price, etc.

        Raises:
            ValueError: If self.cash variable is not a number, then a value error is raised to identify there is an issue. 
            ValueError: If self.cash is less than 2 times the portfolio value, then an error is presented indicating that the system cannot take the next trade due to insufficient cash. This is to be avoided as you should always take the next trade.
            ValueError: If self.cash is less than 0 without leverage, then there is insufficient funds to take the next trade.

        Returns:
            dict: A position is returned consisting of all trade information.
        """         
        price = data['execution_price']
        N = data['N']
        if np.isnan(price) or np.isnan(data['ExL']):
            return position
        if self.shorts:
            if np.isnan(price) or np.isnan(data['ExL']) or np.isnan(data['ExS']):
                return position       
        if np.isnan(N):
            return position 
        dollar_risk = self._get_dollar_risk()
        shares = 0
        if position is None:
            if data['High'] >= data['EL']:
                shares = self._size_position(data, dollar_risk, ticker)
                stop_price = price - (N * self.atr_multiplier)
                long = True
            elif self.shorts:
                if data['Low'] <= data['ES']:
                    shares = self._size_position(data, dollar_risk, ticker)
                    stop_price = price + (N * self.atr_multiplier)
                    long = False
            else:
                return None
            if shares == 0:
                return None
            shares = self._check_cash_balance(shares, price, ticker)
            multiplier_dict = self._check_multipliers()
            multiplier = multiplier_dict.get(ticker,1)
            value = price * shares * multiplier
            self.cash -= value
            if long:
                init_risk = shares * (price - stop_price)
            else:
                init_risk = shares * (stop_price - price)
            position = {
                'shares': shares,
                'entry_price': price,
                'stop_price': stop_price,
                'entry_N': N,
                'init_risk': init_risk,
                'value': value,
                'long':long, 
                'multiplier': multiplier
            }    
            if np.isnan(self.cash):
                raise ValueError(f'Cash is not a number')
            if not ticker.endswith('=F'):
                if self.leverage:
                    if self.cash <= -self.portfolio_value * 2:
                        raise ValueError(f'Ran out of Cash!\n{ticker}\n{data}\n{position}\n{self.cash}\n{-self.portfolio_value*2}')
                else:
                    if self.cash < 0:
                        raise ValueError(f'Ran out of Cash and no leverage!\n{ticker}\n{data}\n{position}\n{self.cash}\n{-self.portfolio_value*2}')
            if self.cash < 0:
                margin_cost = self.daily_interest_rate * abs(self.cash)  # Calculate interest based on the absolute value of negative cash
                self.cash -= margin_cost  # Deduct margin interest from the cash balance
        else:
            if position['long']:
                if data['Low'] <= data['ExL'] or data['Low'] <= position['stop_price']:
                    self.cash += position['shares'] * price * position['multiplier']
                    position = None
            else:
                if data['High'] >= data['ExS'] or data['High'] >= position['stop_price']:
                    self.cash += position['shares'] * price * position['multiplier']
                    position = None
            if position is not None:
                position['value'] = position['shares'] * price * position['multiplier']
                if self.vol_target:
                    if position['long']:
                        rolling_exit_price = max(position['stop_price'],data['ExL'])
                        rolling_risk = (rolling_exit_price - price) * position['shares'] * position['multiplier']
                        rolling_risk_pctEquity = rolling_risk / self.portfolio_value
                        if rolling_risk_pctEquity <= self.risk_target:
                            adjusted_shares = abs(int(np.floor((self.portfolio_value * self.risk_target)/(price - rolling_exit_price))))
                            if adjusted_shares < position['shares']:
                                shares_sold = position['shares'] - adjusted_shares
                                self.cash += shares_sold * price * position['multiplier']
                                position['shares'] = adjusted_shares
                            else:
                                pass
        return position
    
    def run(self) -> None:
        """This runs the whole system and loops through it for every ticker.
        """        
        self.portfolio = {}
        position = {t: None for t in self.tickers}
        for i, (ts,row) in enumerate(self.data.iterrows()):
            for t in self.tickers:
                position[t] = self._run_system(t, row[t], position[t])
            self.portfolio[i] = deepcopy(position)
            self.portfolio[i]['date'] = ts
            self.portfolio[i]['cash'] = copy(self.cash)
            self.portfolio_value = self._calc_portfolio_value(self.portfolio[i])
   
    def get_portfolio_values(self) -> pd.Series:
        """This iterates over the entire portfolio of all markets for the entire straregy and pulls the portfolio values.

        Returns:
            pd.Series: Returns a series of portfolio values for each time period.
        """        
        vals = []
        for v in self.portfolio.values():
            pv = sum([v1['value'] for v0, v1 in v.items() if isinstance(v1, dict)])
            pv += v['cash']
            vals.append(pv)
        return pd.Series(vals, index=self.data.index)
    
    def get_cash_values(self) -> pd.Series:
        """This pulls out all the cash values within the portfolio to monitor cash throughout the strategy.

        Returns:
            pd.Series: Returns a series of cash values. 
        """        
        cash_vals = []
        for v in self.portfolio.values():
            cash_vals.append(v['cash'])
        return pd.Series(cash_vals, index=self.data.index)
    
    def get_init_risk_values(self) -> pd.Series:
        """This pulls out the initial risk values out of the portfolio to monitor the risk that is initially taken.

        Returns:
            pd.Series: Returns a series of initial risk values.
        """        
        init_risk_vals = []
        for v in self.portfolio.values():
            risk = sum([v1['init_risk'] for v0, v1 in v.items() if isinstance(v1, dict)])
            init_risk_vals.append(risk)
        return pd.Series(init_risk_vals, index=self.data.index)