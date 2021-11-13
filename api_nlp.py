import requests
res = requests.post('https://tg-nlp-example-app.herokuapp.com/api/ml', json={"text": 'искать разработчика сайто нужно писать пост инстаграм быть тестовый задание написать небольшой пост тема нужно грамотный написание скопировать рассматривать длительный сотрудничество'})
# if res.ok:
#     print(res.json())
print(res.json())