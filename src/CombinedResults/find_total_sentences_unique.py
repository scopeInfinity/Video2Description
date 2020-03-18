import json, sys
z = json.load(open(sys.argv[1]))
sentences = [x['caption'] for x in z['predicted']]
print("%d Unique sentences out of %d"%(len(set(sentences)),len(sentences)))
