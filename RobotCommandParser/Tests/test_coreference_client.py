import time
import requests

TEST_ONLY_SHORT_COMMANDS = True
SERVICE_URL = "http://127.0.0.1:8893/get_coreference"

commands = []
start_time = time.time()
with open("Data/service_test_examples_coref.txt", "r") as f:
    for line in f:
        commands.append(line)
response = requests.post(SERVICE_URL, json={"commands": commands})
response = response.json()
with open("Data/service_test_examples_coref_response.txt", "w") as f:
    for i in range(len(commands)):
        parse_results_to_line = ""
        for cluster in response["parse_result"][i]:
            clusterLine = "["
            clusterLine += "; ".join([commands[i][m["startPos"]:m["endPos"]] for m in cluster])
            clusterLine += "]"
            parse_results_to_line += clusterLine
        f.write(commands[i].strip()+"\t-\t" + parse_results_to_line +"\n")
print(len(commands), "lines processed in {} seconds".format(time.time()-start_time))

time_deltas = []
with open("Data/service_test_examples_coref.txt", "r") as f:
    for line in f:
        start_time = time.time()
        requests.post(SERVICE_URL, json={"commands": line})
        time_deltas.append(time.time() - start_time)
avg_time = sum(time_deltas) / len(time_deltas)
print("AVG line time to process lines one by one: {}".format(avg_time))