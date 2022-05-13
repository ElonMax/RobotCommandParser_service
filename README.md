# RobotCommandParser_service
Сервис для обработки команд робота, включая предобработку длинных команд. Перенаправляет комменды в другие сервисы.
# Classifier Service
Сервис классификации команд.
api:
- /isready - проводит проверку, загружена ли модель и готов ли сервис к использованию возвращает текст OK и статус 200, если все готово иначе 'Model is not loaded' со статусом 503
- /classify_phrases - Метод обрабатывает post запрос с json данными, содержащими в ключе 'commands' строки с короткими командами. 
Возвращается json c ключем parse_result и значением - списком списков пар из лейблов и классов. Например [[('action', 'move_to'), ('object1', 'house')]]
Настройки сервиса и пути к моделям указываются в Configs/ClassifierService.yml
Запускается просто коммандой
```commandline
python ClassifierService.py
```
Пример обращения в RobotCommandParser/Tests/test_classifier_client.py