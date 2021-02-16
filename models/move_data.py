import os

#os.mkdir("./exp_ER_GNN_Steiner_TSP_50/train/")
#os.mkdir("./exp_ER_GNN_Steiner_TSP_50/train/graph/")
#os.mkdir("./exp_ER_GNN_Steiner_TSP_50/train/solution/")
#os.mkdir("./exp_ER_GNN_Steiner_TSP_50/test/")
#os.mkdir("./exp_ER_GNN_Steiner_TSP_50/test/graph/")
#os.mkdir("./exp_ER_GNN_Steiner_TSP_50/test/solution/")

from os import walk

f = []
for (dirpath, dirnames, filenames) in walk("./exp_ER_GNN_Steiner_TSP_50/"):
    f.extend(filenames)
    break
#print(f)

folder_name = "./exp_ER_GNN_Steiner_TSP_50/"
file_names = []
sol_file_names = []
for elm in f:
 if "graph" in elm:
  file_names.append(folder_name+elm)
  sol_file_names.append(folder_name+"log_folder_exact/"+elm[:-4]+"_output.txt")
#print(file_names)
#print(len(file_names))
print(sol_file_names)

from shutil import copyfile

for i,fname in enumerate(file_names[:-40]):
 src = file_names[i]
 dst = "./exp_ER_GNN_Steiner_TSP_50/train/graph/"+file_names[i].split('/')[-1]
 copyfile(src, dst)
 src = sol_file_names[i]
 dst = "./exp_ER_GNN_Steiner_TSP_50/train/solution/"+sol_file_names[i].split('/')[-1][:-11]+".txt"
 copyfile(src, dst)

for i,fname in enumerate(file_names[-40:]):
 src = file_names[i]
 dst = "./exp_ER_GNN_Steiner_TSP_50/test/graph/"+file_names[i].split('/')[-1]
 copyfile(src, dst)
 src = sol_file_names[i]
 dst = "./exp_ER_GNN_Steiner_TSP_50/test/solution/"+sol_file_names[i].split('/')[-1][:-11]+".txt"
 copyfile(src, dst)



