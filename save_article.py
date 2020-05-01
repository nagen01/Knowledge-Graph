#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 06:56:03 2020

@author: nb01
"""

def putArticleIntoFile(filename, article):
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(article)

article1 = "Apple launches new phone to great reviews. People were stunned at the beautiful new Apple iphones."
article2 = "Apple Watch slammed by critics. There is widespread anger over the lack of security in Apple's Watches. Chinese economy growing at double digits."
article3 = "China economy doing great. It's growing in double digits for the 10th successive quarter."
article4 = "Explosion at Foxconn factory killed 100. An explosion at the Foxconn factory in Shanghai killed a 100 employees totally and desytoyed many plants."

article_list = [article1, article2, article3, article4]        
folder_loc = "/home/nb01/C_Drive/KG/KG_Integration/news/"        

for i in range(len(article_list)):
    filename = folder_loc + str(i) + ".txt"
    putArticleIntoFile(filename, article_list[i])