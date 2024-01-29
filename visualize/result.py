import json
import os
graph_name_set = set()
model_name_set = set()
dataset_name_set = set()
model = {}
for file in os.listdir("result/"):
    text = file.replace("_for_model_"," ").replace("_dataset_"," ").replace(".png","")
    graph_name, model_name, dataset_name = text.split(" ")
    dataset_name_set.add(dataset_name)
    model_name_set.add(model_name)
    graph_name_set.add(graph_name)
    model[dataset_name] = model.get(dataset_name, set())
    model[dataset_name].add(model_name)

for dataset_name in model:
    model[dataset_name] = list(model[dataset_name])

data = {"graphNames": list(graph_name_set), "modelNames": list(model_name_set), "datasetNames": list(dataset_name_set), "model": model}

print(data)
with open("visualize/result.json", "w") as f:
    json.dump(data, f)