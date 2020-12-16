f = open("core.txt", "r")
core = f.read().splitlines()
f.close()
f = open("hardCore.txt", "r")
hardCore = f.read().splitlines()
f.close()
f = open("all.txt", "r")
notCore_tmp = f.read().splitlines()
f.close()
notCore = list()
a = 0
f = open("notCore.txt", "w")
for name in notCore_tmp:
    if name not in core and name not in hardCore:
        a += 1
        print(a, name)
        print(name, file=f)
f.close()
