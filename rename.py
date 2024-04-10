import os

# constable, lionel, lee, va, watts, boudin, cox

# bonheur, boudin, constable, courbet, jones, manet

painter = "manet"

for i, filename in enumerate(os.listdir("paintings/" + painter)):
    print(filename)
    if filename[-5:] == ".jpeg" or filename[-5:] == ".tiff" or filename[-5:] == ".JPEG":
        os.rename("paintings/" + painter + "/" + filename, "paintings/" + painter + "/" + str(i + 1) + filename[-5:])
    else:
        os.rename("paintings/" + painter + "/" + filename, "paintings/" + painter + "/" + str(i + 1) + filename[-4:])