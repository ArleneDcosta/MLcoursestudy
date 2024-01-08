import requests
from bs4 import BeautifulSoup
import csv


if __name__ == '__main__':

    URL = "https://www.topuniversities.com/where-to-study/north-america/united-states/ranked-top-100-us-universities"
    r = requests.get(URL)

    soup = BeautifulSoup(r.content, 'html.parser')
    print(soup.prettify())

    table = soup.find('td')
    print(r.content)


