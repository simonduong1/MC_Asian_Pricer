import pandas as pd
import numpy as np
import yfinance as yf

class DataLoader:
    def __init__(self, ticker, start_date):
        self.ticker = ticker
        self.start_date = pd.to_datetime(start_date)

    def load_data(self):
        df = pd.DataFrame(yf.download(self.ticker, start='2018-01-01'))
        df.index = df.index.tz_localize(None)
        
        last_date_available = df[:(self.start_date - pd.Timedelta(days=1))].index.max()

        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        sigma = df.loc[(self.start_date - pd.DateOffset(years=1)):last_date_available, 'log_returns'].std() * np.sqrt(252)
        mu = df.loc[(self.start_date - pd.DateOffset(years=1)):last_date_available, 'log_returns'].mean() * 252
        S0 = df.loc[df.index == last_date_available, 'Close'].iloc[-1]
        r = self.compute_risk_free_rate(self.start_date)

        return df, S0, sigma, r, last_date_available

    def compute_risk_free_rate(self, start_date):
        url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={start_date.year}'
        rates_table = pd.read_html(url)[0]
        rates_table = rates_table.drop(rates_table.columns[1:10], axis=1)
        rates_table['Date'] = pd.to_datetime(rates_table['Date'])
        return rates_table[rates_table['Date'] < start_date].sort_values(by='Date', ascending=False).iloc[0]['1 Yr'] / 100