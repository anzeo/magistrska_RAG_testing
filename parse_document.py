import re
import requests
from bs4 import BeautifulSoup
import yaml

ai_act_url = 'https://eur-lex.europa.eu/legal-content/SL/TXT/HTML/?uri=OJ:L_202401689'

response = requests.get(ai_act_url)
html_doc = response.text

soup = BeautifulSoup(html_doc, 'html.parser')


def parse_tocke():
    for tocka in soup.find_all('div', id=re.compile(r'rct_\d+')):
        tocka_number, tocka_content = tocka.find_all('p', class_='oj-normal')
        tocka_number = int(tocka_number.text.replace('(', '').replace(')', ''))
        tocka_content = tocka_content.text
        
        data['tocke'].append({
            'id_elementa': tocka.get('id'),
            'tocka': tocka_number,
            'vsebina': tocka_content
        })


def parse_cleni():
    for clen in soup.find_all('div', id=re.compile(r'art_\d+$')):
        # TODO: poišči parent poglavje in po potrebi oddelek (dodaj njuna naslova, itd.)
        parent_poglavje = clen.find_parent('div', id=re.compile(r'cpt_[IVXLCDM]+$'))
        parent_oddelek = clen.find_parent('div', id=re.compile(r'cpt_[IVXLCDM]+\.sct_\d+$'))

        parent_oddelek_data = None
        if parent_oddelek:
            parent_oddelek_data = {
                'id_elementa': parent_oddelek.get('id'),
                'naslov': parent_oddelek.find('div', id=re.compile(r'cpt_[IVXLCDM]+\.sct_\d+\.tit_1$')).find('span').text.replace('\n', ' ').strip()
            }
            # print(parent_oddelek_data)
            # break
        
        parent_poglavje_data = None
        if parent_poglavje:
            parent_poglavje_data = {
                'id_elementa': parent_poglavje.get('id'),
                'naslov': parent_poglavje.find('div', id=re.compile(r'cpt_[IVXLCDM]+\.tit_1$')).find('span').text.replace('\n', ' ').strip()
            }

        clen_number = int(clen.get('id').replace('art_', ''))
        clen_title = clen.find('div', id=re.compile(r'art_\d+\.tit_1$')).find('p').text
        clen_content = ""
        
        for p in clen.find_all('p', class_="oj-normal"):
            if p.parent.name == 'td':
                tr_parent = p.parent.find_parent('tr')
                td_children = tr_parent.find_all('td')
                if td_children.index(p.parent) == 0:
                    # p.parent je prvi child vrstice, torej je oblike (a), (b),... ali (i), (ii),...
                    clen_content += p.get_text(strip=True) + " "
                elif td_children.index(p.parent) == 1:
                    # p.parent je drugi child vrstice, torej vsebuje tekst te vrstice
                    clen_content += p.get_text(strip=True) + "\n"
            else:
                clen_content += p.get_text(strip=True) + "\n"
        
        data['cleni'].append({
            'poglavje': parent_poglavje_data,
            'oddelek': parent_oddelek_data,
            'id_elementa': clen.get('id'),
            'clen': clen_number,
            'naslov': clen_title,
            'vsebina': clen_content.strip()
        })


def write_to_yaml():
    with open('ai_act.yaml', 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)



if __name__ == '__main__':
    data = {'tocke': [], 'cleni': []}
    parse_tocke()
    parse_cleni()
    write_to_yaml()