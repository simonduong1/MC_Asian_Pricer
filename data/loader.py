import pandas as pd
import numpy as np
import yfinance as yf

class DataLoader:
    def __init__(self, ticker, start_date):
        """
        This class contains the methods required to load and clear the data.
        - ticker: the ticker on Yahoo Finance of the underlying asset.
        - start_date: the start date of the option's life.
        """
        self.ticker = ticker
        self.start_date = pd.to_datetime(start_date)

    def load_data(self):
        """
        This method load the data into a pandas' dataframe and compute the parameters needed.
        It returns:
        - df: dataframe containing all the information of YF (Date, Adj Close, Close, High, Low, Open, Volume) and an additional column for log_returns.
        - S0: initial price of the underlying asset.
        - sigma: volatility of the underlying asset.
        - r: continuously compounded risk-free rate.
        - last_date_available: most recent date available in the dataset, prior to the start date of the option's life.
        """
        df = pd.DataFrame(yf.download(self.ticker, start='2018-01-01'))
        df.index = df.index.tz_localize(None)
        
        last_date_available = df[:(self.start_date - pd.Timedelta(days=1))].index.max()

        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        sigma = df.loc[(self.start_date - pd.DateOffset(years=1)):last_date_available, 'log_returns'].std() * np.sqrt(252)
        mu = df.loc[(self.start_date - pd.DateOffset(years=1)):last_date_available, 'log_returns'].mean() * 252
        df.columns = df.columns.droplevel('Ticker')
        S0 = df.loc[df.index == last_date_available, 'Close'].iloc[-1]
        r = self.compute_risk_free_rate(self.start_date)

        return df, S0, sigma, r, last_date_available

    def compute_risk_free_rate(self, start_date):
        """
        This method retrieves the one-year risk-free rate from the US Treasury's website, using the most recent date available prior to the start date.
        - start_date: start date of the option's life.
        """
        url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={start_date.year}'
        rates_table = pd.read_html(url)[0]
        rates_table = rates_table.drop(rates_table.columns[1:10], axis=1)
        rates_table['Date'] = pd.to_datetime(rates_table['Date'])
        return rates_table[rates_table['Date'] < start_date].sort_values(by='Date', ascending=False).iloc[0]['1 Yr'] / 100