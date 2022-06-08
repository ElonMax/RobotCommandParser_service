import time
import requests

SERVICE_URL = "http://127.0.0.1:8895/gapping"

commands = [
    "эти состояния называют фазами воды а превращения из одного состояния в другое фазовыми переходами",
    "описание структуры createstruct приводится ниже а описание структуры mdicreatestruct в разделе, посвященном сообщению wm_mdicreate",
    "иди к лесу, потом к реке",
    "найди человека возле леса потом лес возле человека",
    "иди прямо как дойдешь до леса дельше направо"
]
start_time = time.time()
response = requests.post(SERVICE_URL, json={"commands": commands})
response = response.json()
for i in range(len(commands)):
    parse_results_to_line = response["commands"][i]
print(len(commands), "lines processed in {} seconds".format(time.time()-start_time))

time_deltas = []
for line in commands:
    start_time = time.time()
    result = requests.post(SERVICE_URL, json={"commands": line})
    time_deltas.append(time.time() - start_time)
    print(line)
    print(result.json()["commands"][0])
    print()
avg_time = sum(time_deltas) / len(time_deltas)
print("AVG line time to process lines one by one: {}".format(avg_time))