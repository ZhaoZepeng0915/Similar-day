from pyhive import hive
import pandas as pd
# open connection to hive
conn = hive.connect('112.124.9.195', port=10001)


class GetRawData(object):
    def __init__(self, t_start, t_end):
        self.time_start = t_start
        self.time_end = t_end
        print('time_start:', self.time_start)
        print('time_end:', self.time_end)

    def get_raw_data(self):
        sql = f'SELECT * ' \
              f'from zsjc.silver_zsjc_pivot_1mins ' \
              f'WHERE (date>="{self.time_start}") ' \
              f'and (date<="{self.time_end}") ' \
              f'order by create_time Desc'
        raw_data_df = pd.read_sql(sql, conn)
        raw_data_df.loc[:, 'create_time'] = raw_data_df['create_time'].apply(lambda x: pd.to_datetime(x))
        raw_data_df = raw_data_df.set_index('create_time')
        raw_data_df = raw_data_df.sort_index()
        raw_data_df.to_csv(r'原始数据/zsjc_data_{}_{}.csv'.format(self.time_start, self.time_end), encoding='utf-8')
        return raw_data_df


if __name__ == '__main__':
    # time_start_input = str(input("Please input time_start, eg:2022-06-02:"))
    # time_end_input = str(input("Please input time_end, eg:2022-08-02:"))
    time_start = '2022-08-24'
    time_end = '2022-09-08'
    grd = GetRawData(time_start, time_end)
    grd.get_raw_data()
