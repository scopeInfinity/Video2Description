import ast
a = ast.literal_eval(raw_input().strip())
z=0
for x in a.keys():
	if x[-1] == '3' or x[-1] == '2' or x[-1] == '1':
		continue
	z+=a[x]
print(z)
