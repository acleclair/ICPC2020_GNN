import pickle
import collections

def write(di, filename):
	with open(filename, 'w') as fo:
		for key, val in di.items():
			fo.write("{}, {}\n".format(key, val))

def getdata(filename):
	datas = {}
	for line in open(filename):
		tmp = line.split(',')
		fid = int(tmp[0])
		data = tmp[1].strip()
		datas[fid] = data

	return datas

comsfile = './output/dataset.coms'
tdatsfile = './output/dataset.tdats'
sdatsfile = './output/dataset.sdats'
smlsfile = './output/dataset.srcml'

tvt = pickle.load(open('./output/trainvaltest_ids.pkl', 'rb'))

coms = getdata(comsfile)
tdats = getdata(tdatsfile)
sdats = getdata(sdatsfile)
smls = getdata(smlsfile)

trcoms = {}
trtdats = {}
trsdats = {}
trsmls = {}

for fid in tvt['trainfid']:
        trcoms[fid] = coms[fid]
        trtdats[fid] = tdats[fid]
        trsdats[fid] = sdats[fid]
        trsmls[fid] = smls[fid]

vcoms = {}
vtdats = {}
vsdats = {}
vsmls = {}

for fid in tvt['valfid']:
	vcoms[fid] = coms[fid]
	vtdats[fid] = tdats[fid]
	vsdats[fid] = sdats[fid]
	vsmls[fid] = smls[fid]

tscoms = {}
tstdats = {}
tssdats = {}
tssmls = {}

for fid in tvt['testfid']:
	tscoms[fid] = coms[fid]
	tstdats[fid] = tdats[fid]
	tssdats[fid] = sdats[fid]
	tssmls[fid] = smls[fid]

write(trcoms, './output/coms.train')
write(vcoms, './output/coms.val')
write(tscoms, './output/coms.test')

write(trtdats, './output/tdats.train')
write(vtdats, './output/tdats.val')
write(tstdats, './output/tdats.test')

write(trsdats, './output/sdats.train')
write(vsdats, './output/sdats.val')
write(tssdats, './output/sdats.test')

write(trsmls, './output/smls.train')
write(vsmls, './output/smls.val')
write(tssmls, './output/smls.test')

