from bs4 import BeautifulSoup
import csv

soup = BeautifulSoup(open('DTAB11.htm'))

mylist = []
for name in soup.find_all('p', 'Three-digitLevel'):
    for word in name.strings:
        mylist.append(word.split(" ",1))

num = []
for i in range(0,len(mylist),3):
    num.append(mylist[i][0])

names = []
for j in range(2,len(mylist),3):
    if len(mylist[j]) > 1:
        word = mylist[j][0] + " "+ mylist[j][1]
        names.append(word)
    else:
        names.append(mylist[j][0])

with open('icd9names.csv', 'wb') as f:
    writer = csv.writer(f)
    for word in zip(num,names):
        writer.writerows([word])

with open('icd9names.csv', 'r') as f:
    i = 0
    while i < 50:
        for line in f:
            print line,
        i = i+10
############

list2 = []
for name in soup.find_all('p', 'Fourth-digitLevel'):
    for word in name.strings:
        list2.append(word.split(" ",1))

num = []
for i in range(0,len(list2),3):
    num.append(list2[i][0])

names = []
for j in range(2,len(list2),3):
    if len(list2[j]) > 1:
        word = list2[j][0] + " "+ list2[j][1]
        names.append(word)
    else:
        names.append(list2[j][0])

with open('icd9IndvName.csv', 'wb') as f:
    writer = csv.writer(f)
    for word in zip(num,names):
        writer.writerows([word])

with open('icd9IndvName.csv', 'r') as f:
    i = 0
    while i < 50:
        for line in f:
            print line,
        i = i+10










