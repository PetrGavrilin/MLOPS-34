import pip
 
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])


import_or_install('PyGithub')
from github import Github

import requests

import json
import os
 
# Замените 'YOUR_ACCESS_TOKEN' на свой токен доступа
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN'
 
# Создаем экземпляр объекта класса Github
g = Github(ACCESS_TOKEN)


repo = g.get_repo('PetrGavrilin/UsefulDatasets')
file_content = repo.get_contents('Twitter_volume_AMZN.csv')
download_url = file_content.download_url
response = requests.get(download_url)

with open('/home/petr/project/datasets/data.csv', 'wb') as file:
    file.write(response.content)
