import datetime
import time
import gdax
import pandas as pd
from parameters import *

def main():
    interval = INTERVAL
    data_num = DATA_NUM
    data_batch_size = DATA_BATCH_SIZE

    columns = ["time", "low", "high", "open", "close", "volume"]
    public_client = gdax.PublicClient()

    end_time = int(time.time())
    end_time -= end_time % interval
    gdax_start_time = datetime.datetime(2016,6,1)
    gdax_start_time = time.mktime(gdax_start_time.timetuple())
    if end_time - data_num*interval <= gdax_start_time:
        data_num = (end_time-gdax_start_time)/interval
        data_num = int(data_num)

    cumulative_candles = []
    while len(cumulative_candles) < data_num:
        start_time = end_time - interval * data_batch_size
        print("end time  ", end_time)
        print("start time", start_time)
        st = datetime.datetime.utcfromtimestamp(start_time)
        et = datetime.datetime.utcfromtimestamp(end_time)
        response = public_client.get_product_historic_rates('BTC-USD', granularity=interval, start = st.isoformat(), end = et.isoformat())
        cumulative_candles.extend(response)
        end_time = int(cumulative_candles[-1][0])
        time.sleep(SLEEP_TIME)

    print("Retrieved:", len(cumulative_candles)+1)

    dataframe = pd.DataFrame(cumulative_candles, columns = columns)
    dataframe.to_csv('data.csv', index = False)

if __name__ == '__main__':
    main()
