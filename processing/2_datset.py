import pickle
import re
import collections
import sys

SQL_USERNAME = ''
SQL_PASSWORD = ''


def load(filename):
    return pickle.load(open(filename, 'rb'))

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

def separate_text_struct(dat, reswords):
    textdat = list()
    structdat = list()
    for w in dat:
        if w in reswords:
            structdat.append(w)
            textdat.append('aphcmc') # placeholder token that survives filtering
        else:
            structdat.append('aphcmc')
            textdat.append(w)
    return(' '.join(textdat), ' '.join(structdat))

def camel_case_split_word(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return ' '.join([m.group(0) for m in matches])

def split_camel(tmp, reswords):
    out = list()
    for w in tmp.split(' '):
        if w in reswords:
            out.append(w)
        else:
            out.append(camel_case_split_word(w))
    return(' '.join(out))

def space_special(tmp):
    out = ''
    for w in tmp:
        if w in ['{', '}', '(', ')']:
            out += ' ' + w + ' '
        elif w in ['\n']:
            out += ' '
        else:
            out += w
    return out

def repl_ph(tmp):
    out = ''
    for w in tmp.split(' '):
        if w == 'aphcmc':
            out += ' <ph> '
        else:
            out += ' ' + w + ' '
    return out

import MySQLdb
db = MySQLdb.connect(host='localhost', user=SQL_USERNAME, passwd=SQL_PASSWORD, db='sourcerer')
cur = db.cursor()

def get_context(fid, dats):
    fid = str(fid)
    q = "select F2.id, F2.name from functionalunits F1, functionalunits F2 where F1.fileid=F2.fileid and F1.id="+fid
    cur.execute(q)
    out = list()
    c = 0
    for (ofid, oname) in cur.fetchall():
        if fid == ofid:
            continue
        c += 1
        if c > 50:
            break
        try:
            out.append(dats[int(ofid)].lower())
        except Exception as ex:
            pass # probably just means dats[fid] not found for a function; not a disaster

    return ';'.join(out)   #' '.join(out)

reswordsfd = open('../preprocessing/res_list.txt', 'r')
reswords = set()
for l in reswordsfd:
    l = l.rstrip()
    reswords.add(l)
reswords.add('{')
reswords.add('}')
reswords.add('(')
reswords.add(')')

re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

comfile = './output/dataset.coms'
datsfile = 'fundats-j1.pkl'

coms = collections.defaultdict(str)

try:
    dats = pickle.load(open('output/newdats.pkl', 'rb'))
except Exception as ex:
    dats = load(datsfile)
    newdats = dict()
    c = 0
    for fid, dat in dats.items():
        c += 1
        if c % 100000 == 0:
            print(c)
        newdats[fid] = re_0001_.sub(re_0002, dats[fid])
    dats = newdats
    pickle.dump(dats, open('output/newdats.pkl', 'wb'))

textdats = collections.defaultdict(str)
structdats = collections.defaultdict(str)

c = 0

for line in open(comfile):
    tmp = line.split(',')
    fid = int(tmp[0])
    com = tmp[1]
    
    c += 1
    if c % 100000 == 0:
        print(c)
    


    tmp = dats[fid]


    tmp = tmp.split()
    if len(tmp) > 100:
        continue

    textdat = ' '.join(tmp)
    contdat = get_context(fid, dats)
    textdat = textdat.lower()
    textdats[fid] = textdat
    structdats[fid] = contdat
    coms[fid] = com



outfile1 = './output/dataset.tdats'
outfile2 = './output/dataset.sdats'
new_coms = open('./output/dataset.coms', 'w')
fo = open(outfile1, 'w')
fo2 = open(outfile2, 'w')
for key, val in textdats.items():
    fo.write("{}, {}\n".format(key, val))
    fo2.write("{}, {}\n".format(key, structdats[key]))
    new_coms.write("{}, {}".format(key, coms[key]))

fo.close()
fo2.close()


