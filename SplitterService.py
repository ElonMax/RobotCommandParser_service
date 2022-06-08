import os
from datetime import datetime
import yaml
import logging
from flask import Flask, request, jsonify
from RobotCommandParser.SplitUtils.SplitterWrapper import SplitterWrapper

# Загрузка конфига
with open("Configs/SplitService.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Настройка логгирования
if not os.path.exists("Logs/SplitService"):
    os.mkdir("Logs/SplitService")
logging.basicConfig(format='[%(asctime)s][%(filename)s.%(funcName)s]%(levelname)s: %(message)s',
                    filename="Logs/SplitService/{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                    level=logging.DEBUG if CONFIG["isdebug"] else logging.INFO,
                    datefmt="%Y-%m-%d_%H-%M-%S")

logging.info("Starting Split service")

splitterWrapper = SplitterWrapper(CONFIG)
splitterWrapper.load_model()
# Настройка и запуск flask сервиса
app = Flask("SplitService")

INDEX_PAGE_HELP_TEXT = "<h1>Complex command splitter</h1>" \
                       "<h2>api:</h2>" \
                       "<ul><li>/isready - проводит проверку, загружена ли модель и готов ли сервис к использованию" \
                       "возвращает текст OK и статус 200, если все готово иначе 'Model is not loaded' со статусом 503</li>" \
                       "<li>/split_commands - Метод обрабатывает post запрос с json данными, содержащими в ключе " \
                       "'commands' одну строку. Возвращается json c ключем split_results и значением - списком строк " \
                       "- команд, на которые была разделена команда."

@app.route("/")
def hello_world():
    return INDEX_PAGE_HELP_TEXT


@app.route("/isready")
def isready():
    """
    Проверка, что сервис запущен. настрен. модель подгружена и крутится ожидает данных
    """
    if splitterWrapper.model is not None:
        return "OK", 200
    else:
        return "Model is not loaded", 503


@app.route("/split_commands", methods=['POST'])
def split_command():
    """
    Метод получает на вход json с ключем "commands" по которому находится строка с составной командой
    Возвращает json с ключем split_results, по которому лежит список строк
    """
    result = {"split_results": []}
    if request.is_json:
        json_data = request.get_json()
        logging.debug("Принятый json:{}".format(json_data))
        split_results = splitterWrapper.predict(json_data["commands"])
        result["split_results"] = split_results
        return jsonify(result)
    else:
        raise NotImplementedError("В методе parse_long реализована только обработка json данных. Переданные в POST "
                                  "запрос данные не являются json-ом")
        return jsonify(result), 400


# Запуск сервиса
app.run(host=CONFIG["host"], port=CONFIG["port"], debug=CONFIG["isdebug"])
