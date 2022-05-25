import time
import requests

TEST_ONLY_SHORT_COMMANDS = True
SERVICE_URL = "http://127.0.0.1:8894/split_commands"
time_deltas = []
with open("Data/service_test_examples_split.txt", "r") as inf, open("Data/service_test_examples_split_response.txt", "w") as outf:
    for line in inf:
        start_time = time.time()
        response = requests.post(SERVICE_URL, json={"commands": line})
        time_deltas.append(time.time() - start_time)
        parse_results_to_line = response.json()["split_results"]
        outf.write("INPUT:"+line)
        for subcmd in parse_results_to_line:
            outf.write("\tCMD:"+subcmd)
        outf.write("\n")
avg_time = sum(time_deltas) / len(time_deltas)
print("AVG line time to process lines one by one: {}".format(avg_time))