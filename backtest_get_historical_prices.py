# ============================================================
# INPUT SYMBOLS OR CONIDs AS A LIST
# ============================================================

symbols = [
"FSTA", 
"VDC", 
"XLP", 
"FENY", 
"IYE", 
"VDE", 
"XLE", 
"IYF", 
"VFH", 
"XLF", 
"FHLC", 
"IYH", 
"VHT", 
"XLV", 
"FIDU", 
"XLI", 
"FREL", 
"IYR", 
"VNQ", 
"XLRE", 
"FTEC", 
"IYW", 
"VGT", 
"XLK", 
"FUTY", 
"IDU", 
"VPU", 
"XLU", 
]

# conIds = [320106059, 641561653]

# filename = f"{'_'.join(symbols)}.csv"
filename = "sectors.csv"
output_directory_name = "historical prices"

# ============================================================
# CHOOSE BETWEEN VARIABLE GROUPS
# ============================================================

lookback_period = "5 Y"
length_of_each_period = "1 day"
use_regular_trading_hours = True

# lookback_period = "1 M"
# length_of_each_period = "1 day"
# use_regular_trading_hours = True

prices_to_use = "TRADES"

# ============================================================
# CHOICES:
#    "TRADES"
#    "MIDPOINT"
#    "BID"
#    "ASK"
#    "BID_ASK"
#    "ADJUSTED_LAST"
#    "HISTORICAL_VOLATILITY"
#    "OPTION_IMPLIED_VOLATILITY"
#    "FEE_RATE"
#    "REBATE_RATE"
#    "SCHEDULE"
# ============================================================

# --- imports ---
import asyncio
from datetime import datetime
from pathlib import Path

import pandas as pd
from ib_insync import *


ib = IB()

host = '127.0.0.1'
port = 7496
clientId = int(datetime.now().strftime("%H%M%S"))


async def start_ibkr():
    print(f"Connecting to IBKR host={host}, port={port}, clientId={clientId}")
    if not ib.isConnected():
        await ib.connectAsync(
            host=host,
            port=port,
            clientId=clientId
        )
    print("IBKR connected:", ib.isConnected())
    return ib.isConnected()


async def get_historical_closes_df(contract_list,
                                   lookback_period,
                                   length_of_each_period='1 day',
                                   prices_to_use='TRADES',
                                   use_regular_trading_hours=True):

    df_list = []

    for contract in contract_list:

        sym = contract.symbol

        bars = await ib.reqHistoricalDataAsync(
            contract=contract,
            endDateTime="",          # "" means now
            durationStr=lookback_period,
            barSizeSetting=length_of_each_period,
            whatToShow=prices_to_use,
            useRTH=use_regular_trading_hours,
            formatDate=1
        )

        df = pd.DataFrame([(bar.date, bar.close) for bar in bars], columns=["date", "close"])
        df['date'] = pd.to_datetime(df['date']).dt.date
        df[sym] = df['close']
        df = df.set_index("date")

        df_list.append(df[sym])

    big_df = pd.concat(df_list, axis=1)

    today = pd.Timestamp.today().date()

    if big_df.index[-1] == today:
        big_df = big_df.iloc[:-1]

    big_df = big_df.reindex(sorted(big_df.columns), axis=1)

    return big_df


async def get_historical_prices_df(contract,
                                   lookback_period,
                                   length_of_each_period='1 day',
                                   prices_to_use='TRADES',
                                   use_regular_trading_hours=True):

    bars = await ib.reqHistoricalDataAsync(
        contract=contract,
        endDateTime="",          # "" means now
        durationStr=lookback_period,
        barSizeSetting=length_of_each_period,
        whatToShow=prices_to_use,
        useRTH=use_regular_trading_hours,
        formatDate=1
    )

    df = pd.DataFrame(
        [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]
    )

    df['date'] = pd.to_datetime(df['date']).dt.date

    return df


async def main():

    await start_ibkr()

    try:
        contract_list = []

        for sym in symbols:
        # for conId in conIds:

            contract = Stock(sym, 'SMART', 'USD')
            await ib.qualifyContractsAsync(contract)

            contract_list.append(contract)

            # contract = await ibkr.contract_by_conId(conId)
            # contract = Future(symbol='BRR', lastTradeDateOrContractMonth='202606', exchange='CME', currency='USD')

        df = await get_historical_closes_df(contract_list, lookback_period)

        backtesting_directory = Path.cwd()
        if backtesting_directory.name != "backtesting":
            backtesting_directory = backtesting_directory / "backtesting"
        output_directory = backtesting_directory / output_directory_name

        output_directory.mkdir(parents=True, exist_ok=True)
        output_path = output_directory / filename

        df.to_csv(output_path)
        print(f"Saved {output_path}")
        print("finished")
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
