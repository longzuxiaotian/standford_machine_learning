# -*- coding: utf-8 -*-

import re

def prepare(email):
    with open(email,'r') as edata:
        data = edata.read()
        data = data.lower().replace('$','dollar')
        data = re.sub(r'[0-9]+','number',data)
        data = re.sub(r'http://(.*\.)+[a-z]+( |)','httpaddr ',data)
        data = re.sub(r'([a-z]|-)+@.* *?',' emailaddr',data)
        data = re.sub(r'[^a-z]',' ',data)
        data = re.sub(r' +',' ',data).strip()
        print(data)

if __name__ == '__main__':
    prepare('./file/emailSample2.txt')