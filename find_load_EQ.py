import pandas as pd
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

# Import necessary libraries
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from obspy import UTCDateTime
from roots import get_roots
root_originaldata, root_savefig, root_processeddata = get_roots()

# Specify the IRIS FDSN web service URL
iris_client = Client("IRIS")

def eq_kilauea(min_mag = 3):
    root = root_originaldata
    file_ = pd.read_csv('Kilauea_EQ_201807_3MW.txt', sep=',')

    file = file_[file_.mag > min_mag]
    for i in range(len(file['depth'])):
        if file['depth'][i] < 0:
            file['depth'][i] = 0

    T_lat, T_lon = 19.420908, -155.292023
    model = TauPyModel(model="iasp91")

    file['dist'] = locations2degrees(T_lat, T_lon, file['latitude'], file['longitude'])  # (lat1, long1, lat2, long2)
    arrivaltime = []
    for index, row in file.iterrows():
        arrivaltime.append(model.get_travel_times(source_depth_in_km=abs(row['depth']),
                                                  distance_in_degree=row['dist'])[0].time)
    file['arrivaltime'] = arrivaltime

    return file

# Define the station, network, location, and channel code
network = "HV"
station = "UWE"
location = "--"
channel = "HJ1"

# Define the start and end times for the data request

file = eq_kilauea()

done = ['20180711']

for i in range(len(file['time'])):
    arrivaltime = file['time'][i]
    start_time = UTCDateTime(arrivaltime)
    start_time.hour, start_time.minute, start_time.second = 0,0,0
    end_time = start_time + 24*60*60
    if start_time.month < 10:
        date_name = '%s0%s%s' %(start_time.year, start_time.month, start_time.day)
    else:
        date_name = '%s%s%s' %(start_time.year, start_time.month, start_time.day)

    if date_name in done:
        continue
    try:
        st = iris_client.get_waveforms(network, station, location, channel='HJ1', starttime=start_time, endtime=end_time,
                                        attach_response=True)
        st.select(channel='HJ1').write(root_originaldata + '/Kilauea_%s_HJ1.mseed' % date_name)
        print('HJ1 done')
        st+= iris_client.get_waveforms(network, station, location, channel='HJ2', starttime=start_time, endtime=end_time,
                                        attach_response=True)
        st.select(channel='HJ2').write(root_originaldata + '/Kilauea_%s_HJ2.mseed' % date_name)
        print('HJ2 done')
        st+= iris_client.get_waveforms(network, station, location, channel='HJ3', starttime=start_time, endtime=end_time,
                                        attach_response=True)
        st.select(channel='HJ3').write(root_originaldata + '/Kilauea_%s_HJ3.mseed' % date_name)
        print('HJ3 done')
    except Exception as e:
        print("%s : An error occurred for HJ*:" % arrivaltime, e)

    try:
        st = iris_client.get_waveforms(network, station, location, channel='HNE', starttime=start_time, endtime=end_time,
                                        attach_response=True)
        st.select(channel='HNE').write(root_originaldata + '/Kilauea_%s_HNE.mseed' % date_name)
        print('HNE done')
        st+= iris_client.get_waveforms(network, station, location, channel='HNN', starttime=start_time, endtime=end_time,
                                        attach_response=True)
        st.select(channel='HNN').write(root_originaldata + '/Kilauea_%s_HNN.mseed' % date_name)
        print('HNN done')
        st+= iris_client.get_waveforms(network, station, location, channel='HNZ', starttime=start_time, endtime=end_time,
                                        attach_response=True)
        st.select(channel='HNZ').write(root_originaldata + '/Kilauea_%s_HNZ.mseed' % date_name)


        print("Data downloaded successfully!")
        print(st)
    except Exception as e:
        print("%s : An error occurred for HN*:" % arrivaltime, e)