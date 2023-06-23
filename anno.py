import os

# constable, lionel, lee, va, watts, boudin, cox

painter = 'cox'

path = "annotations/"

f = open(path + painter + ".json", "a")

source = os.listdir("maps/" + painter)
dest = os.listdir("paintings/" + painter)

for  i, src in enumerate(source):
    f.write('{"source": ' + '"maps/' + painter + '/' + src + '", "target": ' + '"paintings/' + painter + '/' + dest[i] + '", "prompt": ""}\n')

f.close()