from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import timedelta

keys = ("name", "locate", "timestamp")
def get_list_of_dict(keys, list_of_tuples): 
    list_of_dict = [dict(zip(keys, values)) for values in list_of_tuples]
    return list_of_dict

# Query Data from InfluxDB
data_range = "1h" # default
def send_puredata(puredata, data_range):
    token = "4DlbT-rE18kBcdF6lSWKOkUFLYf6MYgaVBKQ_G5tkR6317_P3SLJb5fX-oVFVitlFRxfVipeE_DAyIahDfGP6Q=="
    org = "kkuiot"
    bucket = "pun-face"
    url="http://10.101.118.91:8086/"

    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    '''data query filter // can change range like '-5h' for 5 hours, '-20m' for 20 munites, or '-10y' for ten years'''

    flocationGet = query_api.query('from(bucket:"'+str(bucket)+'") |> range(start: -'+data_range+')\
        |> filter(fn:(r) => r._field == "locate" )\
        |> filter(fn:(r) => r._measurement != "Ae" and r._measurement != "Maxwell" )\
        |> timeShift(duration: 7h)')

    # save data into a single list in format list of tuples EX: a = [('kku', 0, 12), ('kok', 1, 13)]
    results = []
    for table in flocationGet:
        for record in table.records:
            results.append((record.get_measurement(), record.get_value(), record.get_time().strftime("%a %d %b %y"), record.get_time().strftime("%X")))
            # tzinfo=tz.gettz('Asia/Bangkok')
    puredata = results
    return puredata

def send_dictdata(dictdata, data_range):
    token = "4DlbT-rE18kBcdF6lSWKOkUFLYf6MYgaVBKQ_G5tkR6317_P3SLJb5fX-oVFVitlFRxfVipeE_DAyIahDfGP6Q=="
    org = "kkuiot"
    bucket = "pun-face"
    url="http://10.101.118.91:8086/"

    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    '''data query filter // can change range like '-5h' for 5 hours, '-20m' for 20 munites, or '-10y' for ten years'''

    flocationGet = query_api.query('from(bucket:"'+str(bucket)+'") |> range(start: -'+data_range+')\
        |> filter(fn:(r) => r._field == "locate" )\
        |> timeShift(duration: 7h)')

    # save data into a single list in format list of tuples EX: a = [('kku', 0, 12), ('kok', 1, 13)]
    results = []
    for table in flocationGet:
        for record in table.records:
            results.append((record.get_measurement(), record.get_value(), record.get_time().strftime("%c")))
            # tzinfo=tz.gettz('Asia/Bangkok')
    dictdata = get_list_of_dict(keys, results) 
    return dictdata