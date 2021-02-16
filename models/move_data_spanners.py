import os

id_file_name = "/Users/abureyanahmed/Graph_spanners/exp_ER_wgt_20/id_to_file.csv"
experiment_folder = "/Users/abureyanahmed/Graph_spanners/exp_ER_wgt_20/"
folder_name = "./exp_all_GNN_spanner/"
'''
os.mkdir(folder_name)
os.mkdir(folder_name+"train/")
os.mkdir(folder_name+"train/graph/")
os.mkdir(folder_name+"train/solution/")
os.mkdir(folder_name+"test/")
os.mkdir(folder_name+"test/graph/")
os.mkdir(folder_name+"test/solution/")
quit()
'''

def parse_id_csv(experiment_folder, graph_type):
  ids = []
  folders = []
  stretch = []
  file_names = []
  level = []
  nlevel = []
  node = []
  f = open(experiment_folder + 'id_to_file.csv', 'r')
  line_number = 1
  while True:
   line = f.readline()
   if line=='':
    break
   arr = line.split(';')
   FILE_NAME = arr[3]
   if not graph_type in FILE_NAME:
    continue
   file_names.append(FILE_NAME)
   ids.append(arr[0])
   CODE_FILE = arr[1]
   ROOT_FOLDER = arr[2]
   folders.append(ROOT_FOLDER)
   STRETCH_FACTOR = float(arr[4])
   stretch.append(str(STRETCH_FACTOR))
   line_number += 1
  f.close()
  for i in range(len(folders)):
   node.append(int(file_names[i].split('_')[5]))
   level.append(file_names[i].split('_')[4])
   nlevel.append(int(file_names[i].split('_')[3]))
  return ids, folders, stretch, file_names, level, node, nlevel

from os import walk

f1 = []
f2 = []
#for (dirpath, dirnames, filenames) in walk("./exp_ER_GNN_Steiner_TSP_50/"):
#    f.extend(filenames)
#    break
ids, folders, stretch, file_names, level, node, nlevel = parse_id_csv(experiment_folder, "ER")
for i, nl in enumerate(nlevel):
  #if nl==1:
  if level[i]=='E':
    f1.append(file_names[i])
  else:
    f2.append(file_names[i])
f = f1[:-40] + f2[:-40] + f1[-40:] + f2[-40:]
#print(f)
#print(len(f))
#quit()

#folder_name = "./exp_ER_GNN_Steiner_TSP_50/"
file_names = []
sol_file_names = []
for elm in f:
 if "graph" in elm:
  file_names.append(experiment_folder+elm+".txt")
  sol_file_names.append(experiment_folder+"log_folder/"+elm+"_output.txt")
#print(file_names)
#print(len(file_names))
#print(sol_file_names)
#quit()

from shutil import copyfile

for i,fname in enumerate(file_names[:-80]):
 src = file_names[i]
 dst = folder_name+"train/graph/"+file_names[i].split('/')[-1]
 copyfile(src, dst)
 src = sol_file_names[i]
 #dst = folder_name+"train/solution/"+sol_file_names[i].split('/')[-1][:-11]+".txt"
 dst = folder_name+"train/solution/"+sol_file_names[i].split('/')[-1]
 copyfile(src, dst)

for i,fname in enumerate(file_names[-80:]):
 src = file_names[i]
 dst = folder_name+"test/graph/"+file_names[i].split('/')[-1]
 copyfile(src, dst)
 src = sol_file_names[i]
 #dst = folder_name+"test/solution/"+sol_file_names[i].split('/')[-1][:-11]+".txt"
 dst = folder_name+"test/solution/"+sol_file_names[i].split('/')[-1]
 copyfile(src, dst)



