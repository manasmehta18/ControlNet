import json
import random

with open("annotations/all.json") as f:
     data = json.load(f)

random.shuffle(data)

with open("annotations/all.json") as f:
     json.dump(data, f)