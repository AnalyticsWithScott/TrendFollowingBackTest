# TrendFollowing Trading Strategy Back Test

## Overview
This project implements a trend-following trading strategy using Python. The strategy is based on Donchian Channels and Moving Averages, with additional risk management features such as position sizing, stop loss, and leverage control. Additionally, the python code uses the YFinance API to fetch data and this code runs across an entire portfolio of markets.

## Features
- **Donchian Channels:** Identifies breakouts for entering and exiting positions.
- **Moving Averages:** Uses fast and slow moving averages to determine the trend direction.
- **Risk Management:** Implements position sizing based on risk percentage per trade and stop loss orders using Average True Range (ATR).
- **Leverage Control:** Allows for turning on or off leverage/margin.
- **Performance Metrics:** Calculates various performance metrics such as total returns, annualized returns, volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and more.
