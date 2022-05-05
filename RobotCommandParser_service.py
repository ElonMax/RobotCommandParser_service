import os
from datetime import datetime
import yaml
import logging
from flask import Flask, request, jsonify
import requests

# Загрузка конфига
with open("Configs/RobotCommandParser_service.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Настройка логгирования
if not os.path.exists("Logs/RobotCommandParser_service"):
    os.mkdir("Logs/RobotCommandParser_service")
logging.basicConfig(format='[%(asctime)s] /%(filename)s.%(funcName)s/ %(levelname)s: %(message)s',
                    filename="Logs/RobotCommandParser_service/{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                    level=logging.DEBUG if CONFIG["isdebug"] else logging.INFO,
                    datefmt="%Y-%m-%d_%H-%M-%S")

logging.info("Starting main RobotCommandParser service")

app = Flask("RobotCommandParser_service")

INDEX_PAGE_HELP_TEXT = "<h1>Hello, from Robot command parser service.</h1>" \
                       "<h2>api:</h2>" \
                       "<ul><li>/parse_long - Метод обрабатывает post запрос с json данными, содержащими в ключе " \
                       "'command' строку с командой. Обработка длинной команды, включает разметку на кореференцию, " \
                       "разбивку на мелкие команды, классификацию всех малых команд и возвращает агрегацию классификации." \
                       "Возвращается json c ключем parse_result и значением - списком со списками пар из лейблов и " \
                       "классов. Например [(action, move_to), (object1, house)]</li>" \
                       "<li>/parse_short - Метод обрабатывает post запрос с json данными, содержащими в ключе " \
                       "'command' строку с командой. Строка будет обрабатываться как короткая команда. Т.е. " \
                       "отправляется сразу в мультилейбл классификатор для определения шаблона и заполнения его классами." \
                       "Возвращается json c ключем parse_result и значением - списоком пар из лейблов и классов. " \
                       "Например [(action, move_to), (object1, house)]</li></ul>"


@app.route("/")
def hello_world():
    return INDEX_PAGE_HELP_TEXT


@app.route("/check_other_servies")
def check_other_servies():
    """
    Отправить на запрос на внутренние сервисы разметки, спросить, готовы ли они.
    :return:
    """
    raise NotImplementedError("Метод проверки ещё не реализован")
    return "", 503


@app.route("/parse_long", methods=['POST'])
def parse_long():
    """
    Метод обрабатывает post запрос с json данными, содержащими в ключе "command" строку с командой.
    Обработка длинной команды, включает разметку на кореференцию, разбивку на мелкие команды,
    классификацию всех малых команд и возвращает агрегацию классификации.
    :return:
    Возвращается json c ключем parse_result и
    значением - списком со списками пар из лейблов и классов. Например [(action, move_to), (object1, house)]
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


@app.route("/parse_short", methods=['POST'])
def parse_short():
    """
    Метод обрабатывает post запрос с json данными, содержащими в ключе "command" строку с командой.
    Строка будет обрабатываться как короткая команда. Т.е. отправляется сразу в мультилейбл классификатор для
    определения шаблона и заполнения его классами.
    :return:
    Возвращается json c ключем parse_result и
    значением - списоком пар из лейблов и классов. Например [(action, move_to), (object1, house)]
    """
    result = {"parse_result": []}
    if request.is_json:
        json_data = request.get_json()
        logging.debug("Принятый json:{}".format(json_data))
        classification_results = requests.post(CONFIG["classifier_url"], json={"commands": [json_data["command"]]})
        result["parse_result"] = classification_results
        return jsonify(result)
    else:
        raise NotImplementedError("В методе parse_short реализована только обработка json данных. Переданные в POST "
                                  "запрос данные не являются json-ом")
        return jsonify(result), 400


# Запуск сервиса
app.run(host=CONFIG["host"], port=CONFIG["port"], debug=CONFIG["isdebug"])
