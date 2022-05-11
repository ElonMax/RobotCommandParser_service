import time
import requests

TEST_ONLY_SHORT_COMMANDS = True
SERVICE_URL = "http://127.0.0.1:8892/classify_phrases"

commands = []
start_time = time.time()
with open("Data/service_test_examples_short.txt", "r") as f:
    for line in f:
        commands.append(line)
response = requests.post(SERVICE_URL, json={"commands": commands})
response = response.json()
with open("Data/service_test_examples_short_response.txt", "w") as f:
    for i in range(len(commands)):
        parse_results_to_line = response["parse_result"][i]
        parse_results_to_line = "; ".join(["{}:{}".format(l, c) for l, c in parse_results_to_line])
        f.write(commands[i].strip()+"\t-\t" + parse_results_to_line +"\n")
print(len(commands), "lines processed in {} seconds".format(time.time()-start_time))

time_deltas = []
with open("Data/service_test_examples_short.txt", "r") as f:
    for line in f:
        start_time = time.time()
        requests.post(SERVICE_URL, json={"commands": line})
        time_deltas.append(time.time() - start_time)
avg_time = sum(time_deltas) / len(time_deltas)
print("AVG line time to process lines one by one: {}".format(avg_time))