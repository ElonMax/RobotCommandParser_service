import os
from datetime import datetime
import yaml
import logging
from flask import Flask, request, jsonify
from RobotCommandParser.CoreferenceUtils.CoreferenceWrapper import CoreferenceWrapper

# Загрузка конфига
with open("Configs/CoreferenceService.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Настройка логгирования
if not os.path.exists("Logs/CoreferenceService"):
    os.mkdir("Logs/CoreferenceService")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(format='[%(asctime)s][%(filename)s.%(funcName)s]%(levelname)s: %(message)s',
                    filename="Logs/CoreferenceService/{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                    level=logging.DEBUG if CONFIG["isdebug"] else logging.INFO,
                    datefmt="%Y-%m-%d_%H-%M-%S")
print("Logs/CoreferenceService/{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
logging.info("Starting coreference service")

coreferenceWrapper = CoreferenceWrapper(CONFIG)
coreferenceWrapper.load_model()
# Настройка и запуск flask сервиса
app = Flask("CoreferenceService")

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


@app.route("/get_coreference", methods=['POST'])
def get_coreference():
    """
    Метод получает на вход json с ключем "commands" по которому находится список строк - команд и отправляет
    в модель разрешения кореференции
    Возвращает json с ключем parse_result, по которому для каждой фразы лежит список кластеров, каждый кластер - список
    словарей упоминаний с ключами startPos и endPos, указывающими начало и конец сущности в строке.
    """
    result = {"parse_result": []}
    if request.is_json:
        json_data = request.get_json()
        logging.debug("Принятый json:{}".format(json_data))
        try:
            parse_results = coreferenceWrapper.predict(json_data["commands"])
        except:
            parse_results = [[] for phrase in json_data["commands"]]
        result["parse_result"] = parse_results
        return jsonify(result)
    else:
        raise NotImplementedError("В методе parse_long реализована только обработка json данных. Переданные в POST "
                                  "запрос данные не являются json-ом")


# Запуск сервиса
app.run(host=CONFIG["host"], port=CONFIG["port"], debug=CONFIG["isdebug"])
