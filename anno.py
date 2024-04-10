import os

# constable, lionel, lee, va, watts, boudin, cox

# bonheur = 1, boudin = 2, constable = 3, courbet = 4, jones = 5, manet = 6

painter = 'manet'

path = "annotations/"

f = open(path + painter + ".json", "a")

source = os.listdir("maps/" + painter)
dest = os.listdir("paintings/" + painter)

for  i, des in enumerate(dest):
    print(des)
    f.write('{"source": ' + '"maps/' + painter + '/' + des.rsplit( ".", 1 )[ 0 ] + '.jpg' + '", "target": ' + '"paintings/' + painter + '/' + des + '", "prompt": "a 19th-century Realist oil painting of cities, the ocean, beaches, people, and clouds by the artist ' + painter + '", "art": "6"}\n')

f.close()