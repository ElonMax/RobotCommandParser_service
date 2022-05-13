import os
from datetime import datetime
import yaml
import logging
from flask import Flask, request, jsonify
from RobotCommandParser.ClassificationUtils.ClassifierWrapper import ClassifierWrapper

# Загрузка конфига
with open("Configs/ClassifierService.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Настройка логгирования
if not os.path.exists("Logs/CommandClassifier"):
    os.mkdir("Logs/CommandClassifier")
logging.basicConfig(format='[%(asctime)s][%(filename)s.%(funcName)s]%(levelname)s: %(message)s',
                    filename="Logs/CommandClassifier/{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                    level=logging.DEBUG if CONFIG["isdebug"] else logging.INFO,
                    datefmt="%Y-%m-%d_%H-%M-%S")

logging.info("Starting CommandClassifier service")

classifierWrapper = ClassifierWrapper(CONFIG)
classifierWrapper.load_model()
# Настройка и запуск flask сервиса
app = Flask("ClassifierService")

INDEX_PAGE_HELP_TEXT = "<h1>Short command classifier</h1>" \
                       "<h2>api:</h2>" \
                       "<ul><li>/isready - проводит проверку, загружена ли модель и готов ли сервис к использованию" \
                       "возвращает текст OK и статус 200, если все готово иначе 'Model is not loaded' со статусом 503</li>" \
                       "<li>/classify_phrases - Метод обрабатывает post запрос с json данными, содержащими в ключе " \
                       "'commands' строки с короткими командами." \
                       "Возвращается json c ключем parse_result и значением - списком списков пар из лейблов и классов. " \
                       "Например [[('action', 'move_to'), ('object1', 'house')]]</li></ul>"

@app.route("/")
def hello_world():
    return INDEX_PAGE_HELP_TEXT


@app.route("/isready")
def isready():
    """
    Проверка, что сервис запущен. настрен. модель подгружена и крутится ожидает данных
    """
    if classifierWrapper.model is not None:
        return "OK", 200
    else:
        return "Model is not loaded", 503


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
        classification_results = classifierWrapper.predict(json_data["commands"])
        result["parse_result"] = classification_results
        return jsonify(result)
    else:
        raise NotImplementedError("В методе parse_long реализована только обработка json данных. Переданные в POST "
                                  "запрос данные не являются json-ом")
        return jsonify(result), 400


# Запуск сервиса
app.run(host=CONFIG["host"], port=CONFIG["port"], debug=CONFIG["isdebug"])
