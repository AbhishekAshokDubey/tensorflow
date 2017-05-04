# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:22:55 2017

@author: ADubey4
"""

#links:
#    https://pymotw.com/2/getopt/
#https://scrapy.org/

import sys, os, getopt
import requests
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
import pandas as pd
import queue
import re;

os.chdir(r"C:\Users\Adubey4\Desktop\chatbot\code")

#from lxml import html
#tree = html.fromstring(reponse.content)

def get_arg(argv):
    try:
#        opts, args = getopt.getopt(argv, '',['url=', 'max_hop='])
        opts, args = getopt.getopt(['--url', "https://en.wikipedia.org/wiki/SMS_Kaiser_Barbarossa", '--max_hop', 1], '',['url=', 'max_hop='])
    except:
        print('readweb.py --url <web_url> --max_hop <hop_count>')
        sys.exit(2)
    arg_dict = {}
    for var, val in opts:
        arg_dict[var.replace("--","")] = val
    if arg_dict.get('url', None) == None:
        arg_dict['url'] = "https://en.wikipedia.org/wiki/Main_Page"
    if arg_dict.get('max_hop', None) == None:
        arg_dict['max_hop'] = 1
    elif arg_dict.get('max_hop', None) == '':
        arg_dict['max_hop'] = 0
    else:
        arg_dict['max_hop'] = int(arg_dict['max_hop'])
    return arg_dict;


def get_text_links(url, hop_no, max_hop = 2):
    print(url)
    if hop_no > max_hop:
        return [], [];
    else:
        text_list = []
        links_list = []
        reponse = requests.get(url)
        if reponse.status_code == 200:
            pattern = re.compile('[\W_]+', re.UNICODE)
            soup = bs(reponse.text, "lxml")
            paras = soup.findAll('p')
            anchor_tags = soup.findAll('a')
            
            for para in paras:
                text_list.append(pattern.sub(' ', para.text))

            for link in anchor_tags:
                if link.has_attr('href'):
#                    print(hop_no)
                    links_list.append(urljoin(url, link['href']))
        return " ".join(text_list), links_list


if __name__ == "__main__":
    all_text = []
    arg_dict = get_arg(sys.argv[1:])
#    arg_dict = get_arg("lala")
    max_hop = arg_dict['max_hop']

    all_links = set();
    pending_links_queue = queue.Queue();
    
    all_links.add(arg_dict['url'])
    pending_links_queue.put((arg_dict['url'], 0))

    while pending_links_queue.qsize() > 0:
        link, hop = pending_links_queue.get_nowait();
        text_list, links_list = [], []
        if hop <= max_hop:
            text_list, links_list = get_text_links(link, hop, max_hop)
            all_text.append(text_list)
            if hop+1 <= max_hop:
                for link in links_list:
                    if link not in all_links:
                        all_links.add(link)
                        pending_links_queue.put_nowait((link, hop+1))

    text_df = pd.DataFrame(columns=["text"])
    text_df["text"] = all_text
    text_df.to_csv("wiki.csv")