import os
from datetime import datetime
import yaml
import logging
from flask import Flask, request, jsonify
from RobotCommandParser.GappingUtils.GappingWrapper import GappingWrapper

# Загрузка конфига
with open("Configs/GappingService.yml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Настройка логгирования
if not os.path.exists("Logs/GappingService"):
    os.mkdir("Logs/GappingService")
logging.basicConfig(format='[%(asctime)s][%(filename)s.%(funcName)s]%(levelname)s: %(message)s',
                    filename="Logs/GappingService/{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
                    level=logging.DEBUG if CONFIG["isdebug"] else logging.INFO,
                    datefmt="%Y-%m-%d_%H-%M-%S")

logging.info("Starting Gapping service")

gappingWrapper = GappingWrapper(CONFIG["Wrapper"])
gappingWrapper.load_model()
# Настройка и запуск flask сервиса
app = Flask("GappingService")

INDEX_PAGE_HELP_TEXT = "<h1>Gapping resolution service</h1>" \
                       "<h2>api:</h2>" \
                       "<ul><li>/isready - проводит проверку, загружена ли модель и готов ли сервис к использованию" \
                       "возвращает текст OK и статус 200, если все готово иначе 'Model is not loaded' со статусом 503</li>" \
                       "<li>/gapping - Метод обрабатывает post запрос с json данными, содержащими в ключе " \
                       "'commands' одну строку. Возвращается json c ключем commands, с изменённым значением"

@app.route("/")
def hello_world():
    return INDEX_PAGE_HELP_TEXT


@app.route("/isready")
def isready():
    """
    Проверка, что сервис запущен. настрен. модель подгружена и крутится ожидает данных
    """
    if gappingWrapper.model is not None:
        return "OK", 200
    else:
        return "Model is not loaded", 503


@app.route("/gapping", methods=['POST'])
def resolve_gapping():
    """
    Метод получает на вход json с ключем "commands" по которому находится строка с составной командой
    Возвращает json с ключем commands, по которому лежит тоже строка
    """
    result = {"commands": []}
    if request.is_json:
        json_data = request.get_json()
        logging.debug("Принятый json:{}".format(json_data))
        gap_resolution_results = gappingWrapper.predict(json_data["commands"])
        result["commands"] = gap_resolution_results
        return jsonify(result)
    else:
        raise NotImplementedError("В методе parse_long реализована только обработка json данных. Переданные в POST "
                                  "запрос данные не являются json-ом")
        return jsonify(result), 400


# Запуск сервиса
app.run(host=CONFIG["host"], port=CONFIG["port"], debug=CONFIG["isdebug"])
