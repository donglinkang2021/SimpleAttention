import json
import os
graph_name_set = set()
model_name_set = set()
dataset_name_set = set()
for file in os.listdir("result/"):
    text = file.replace("_for_model_"," ").replace("_dataset_"," ").replace(".png","")
    graph_name, model_name, dataset_name = text.split(" ")
    graph_name_set.add(graph_name)
    model_name_set.add(model_name)
    dataset_name_set.add(dataset_name)
# print(f"const graphOptions = {list(graph_name_set)};\n")
# print(f"const modelOptions = {list(model_name_set)};\n")
# print(f"const datasetOptions = {list(dataset_name_set)};\n")
data = {}
data["graphNames"] = list(graph_name_set)
data["modelNames"] = list(model_name_set)
data["datasetNames"] = list(dataset_name_set)
with open("visualize/result.json", "w") as f:
    json.dump(data, f)