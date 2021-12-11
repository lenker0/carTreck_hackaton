# carTreck_hackaton
Перед началом аренды пользователь делает селфи в приложении. Нужно разработать программу, с помощью которой, имея фото или видео с фронтальной камеры, можно достоверно сказать, это фото “живого” человека, или же нет (фото с экрана смартфона, фото в маске, дипфейк и т.д


## Environment
Чтобы установить окружение выполните следующие команды в консоли
### Важно, чтобы версия python должна быть 3.7 - 3.9
```
git clone https://github.com/lenker0/carTreck_hackaton.git
cd carTreck_hackaton
pip install -r requirements.txt
```

## Run
Для того, чтобы запустить веб-приложение, которое из-за отсутствия сервера, пока что работает без вебки :) (я попробовал ngrok, чтобы вы смогли зайти на сайт, но он включит мою камеру, а не камеру клиента, который зайдет на сайт)
Но мы расскажем как действовать без нее))

###Программа сработает только тогда, когда заметит в камере лицо (не важно, реальное или фейк)

### Для начала откройте *double click'ом*
```
start.bat
```
### Затем перейдите на локалхост
```
http://127.0.0.1:5000/
(может отличаться, проверьте в консоле)
```
### Сядьте поудобнее, нажмите кнопку *"Проверить"* и получите результат нашей программы
Для повтороного селфи нажмите 
```
CTRL + SHIFT + R
```
