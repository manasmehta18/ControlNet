import os

# constable, lionel, lee, va, watts, boudin, cox

# bonheur = 1, boudin = 2, constable = 3, courbet = 5, jones = 6, manet = 4

painter = 'manet'

path = "annotations/sky/"

f = open(path + painter + ".json", "a")

source = os.listdir("maps/" + painter)
dest = os.listdir("paintings/" + painter)

for  i, des in enumerate(dest):
    print(des)
    f.write('{"source": ' + '"maps/' + painter + '/' + des.rsplit( ".", 1 )[ 0 ] + '.jpg' + '", "target": ' + '"paintings/' + painter + '/' + des + '", "prompt": "a 19th-century Realist oil painting of clouds and the sky by the artist ' + painter + '", "art": "4"}\n')

f.close()