import os
from datetime import datetime
import yaml
import logging
from flask import Flask, request, jsonify

# Загрузка конфига
with open("Configs/ClassifierService.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Настройка логгирования
if not os.path.exists("Logs/CommandClassifier"):
    os.mkdir("Logs/CommandClassifier")
logging.basicConfig(format='[%(asctime)s] /%(filename)s.%(funcName)s/ %(levelname)s: %(message)s',
                    filename="Logs/CommandClassifier/{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                    level=logging.DEBUG if CONFIG["isdebug"] else logging.INFO,
                    datefmt="%Y-%m-%d_%H-%M-%S")

logging.info("Starting CommandClassifier service")

# Настройка и запуск flask сервиса
app = Flask("ClassifierService")

INDEX_PAGE_HELP_TEXT = "<h1>Short command classifier</h1>"


@app.route("/")
def hello_world():
    return INDEX_PAGE_HELP_TEXT


@app.route("/isready")
def isready():
    """
    Проверка, что сервис запущен. настрен. модель подгружена и крутится ожидает данных
    """
    raise NotImplementedError("Метод проверки ещё не реализован")
    return "", 503


@app.route("/classify_phrases", methods=['POST'])
def classify_phrases():
    """
    Метод получает на вход json с ключем "commands" по которому находится список коротких команд и отправляет
    в классификатор
    Возвращает json с ключем parse_result, по которому лежит список со списками пар из лейблов и классов
    """
    result = {"parse_result": []}
    if request.is_json:
        json_data = request.get_json()
        logging.debug("Принятый json:{}".format(json_data))
        return jsonify(result)
    else:
        raise NotImplementedError("В методе parse_long реализована только обработка json данных. Переданные в POST "
                                  "запрос данные не являются json-ом")
        return jsonify(result), 400


# Запуск сервиса
app.run(host=CONFIG["host"], port=CONFIG["port"], debug=CONFIG["isdebug"])
