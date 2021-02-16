from os import walk
import numpy as np
import tensorflow
import networkx as nx
import matplotlib.pyplot as plt

# read graph

def is_comment(x):
 if x[0]=='#':
  return True
 return False

def take_input(input_file, negate_one=True):
 file = open(input_file,"r")
 #print("File name: "+input_file)
 while True:
  l = file.readline()
  #print(l)
  if not is_comment(l):
   break
 m = int(l)
 edge_list = list()
 for i in range(m):
    while True:
     l = file.readline()
     if len(l) == 0:
      break
     if not is_comment(l):
      break
    t_arr1 = []
    t_arr2 = l.split()
    if(len(t_arr2)<3):break
    if negate_one:
     t_arr1.append(int(t_arr2[0])-1)
     t_arr1.append(int(t_arr2[1])-1)
    else:
     t_arr1.append(int(t_arr2[0]))
     t_arr1.append(int(t_arr2[1]))
    #t_arr1.append(int(t_arr2[2]))
    t_arr1.append(float(t_arr2[2]))
    edge_list.append(t_arr1)

 levels = int(file.readline())
 tree_ver=[]
 #tree_ver = [(int(x)-1) for x in raw_input().split()]
 for l in range(levels):
  #print "Steiner tree vertices of level "+str(l+1)+":"
  if negate_one:
   tree_ver.append([(int(x)-1) for x in file.readline().split()])
  else:
   tree_ver.append([(int(x)) for x in file.readline().split()])

 file.close()
 return edge_list, tree_ver

def get_graph_info(train_graph_folder, train_solution_folder):
 f = []
 for (dirpath, dirnames, filenames) in walk(train_graph_folder):
  f.extend(filenames)
  break
 train_edges, train_solutions, train_terminals = [], [], []
 for file in f:
  #print("file name", file)
  graph_edges, terminals = take_input(train_graph_folder+file)
  solution_edges, terminals = take_input(train_solution_folder+file[:-4]+"_output.txt", False)
  train_edges.append(graph_edges)
  train_solutions.append(solution_edges)
  train_terminals.append(terminals)
 #print("train_edges[0]", train_edges[0])
 return train_edges, train_solutions, train_terminals

def ordered_edge(u, v):
  if u>v:
    t = u
    u = v
    v = t
  return u, v

def edge_to_number(labels = 'nodes'):
 #max_node_size = 20
 max_node_size = 50
 edge_to_id = dict()
 id_to_edge = dict()
 idx = 0
 for u in range(max_node_size):
  for v in range(u+1, max_node_size):
   edge_to_id[(u, v)] = idx
   id_to_edge[idx] = (u, v)
   idx = idx + 1
 n_inputs = idx + max_node_size
 n_outputs = max_node_size
 if labels == 'nodes':
  return n_inputs, n_outputs, idx, edge_to_id, id_to_edge
 else:
  return n_inputs, idx, idx, edge_to_id, id_to_edge

def generate_dataset(train_graph_edges, train_terminals, train_solution_edges, labels = 'nodes'):
 n_inputs, n_outputs, max_edge_index, edge_to_id, id_to_edge = edge_to_number(labels)
 train_features = []
 train_labels = []
 for i, graph_edges in enumerate(train_graph_edges):
  #print("graph_edges", graph_edges)
  feature_arr = [0 for j in range(n_inputs)]
  label_arr = [0 for j in range(n_outputs)]
  for e in graph_edges:
   u, v, w = e
   u, v = ordered_edge(u, v)
   feature_arr[edge_to_id[(u, v)]] = w
  terminals = train_terminals[i]
  for t in terminals[0]:
   feature_arr[max_edge_index+t] = 1
  solution_edges = train_solution_edges[i]
  for e in solution_edges:
   #for e in solution_edges:
    u, v, w = e
    u, v = ordered_edge(u, v)
    if labels=='nodes':
     #'''
     label_arr[u] = 1
     label_arr[v] = 1
     #'''
     '''
     if u not in terminals[0]:
      label_arr[u] = 1
     if v not in terminals[0]:
      label_arr[v] = 1
     '''
    else:
     label_arr[edge_to_id[(u, v)]] = 1
   
  train_features.append(feature_arr)
  train_labels.append(label_arr)

 train_features = np.vstack(train_features)
 train_labels = np.vstack(train_labels)
 return train_features, train_labels, n_inputs, n_outputs

def binary_accuracy(system: np.ndarray, human: np.ndarray):
    return np.mean(np.round(system) == human)

def precision_recall(y_hat, y_actual):
    y_actual = y_actual.flatten()
    y_hat = np.round(y_hat).flatten()
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    print("TP, FP, TN, FN", TP, FP, TN, FN)
    if (TP+FP)==0:
     prec = 1
    else:
     prec = TP/(TP+FP)
    print("Precision", prec)
    if (TP+FN)==0:
     recall = 1
    else:
     recall = TP/(TP+FN)
    print("Recall", recall)
    print("F score", 2*prec*recall/(prec+recall))
    return(TP, FP, TN, FN)

def load_data(folder_name):
 #folder_name = "/Users/abureyanahmed/GNN/TSP_Steiner/Steinertree-master/steiner_dataset/"
 #folder_name = "/Users/abureyanahmed/GNN/TSP_Steiner/Steinertree-master/steiner_dataset50/"
 train_graph_folder = folder_name + "train/graph/"
 train_solution_folder = folder_name + "train/solution/"
 test_graph_folder = folder_name + "test/graph/"
 test_solution_folder = folder_name + "test/solution/"

 #print("Getting train data")
 train_graph_edges, train_solution_edges, train_terminals = get_graph_info(train_graph_folder, train_solution_folder)
 #print("Getting test data")
 test_graph_edges, test_solution_edges, test_terminals = get_graph_info(test_graph_folder, test_solution_folder)

 return train_graph_edges, train_solution_edges, train_terminals, test_graph_edges, test_solution_edges, test_terminals

'''
folder_name = "/Users/abureyanahmed/GNN/TSP_Steiner/Steinertree-master/steiner_dataset/"
train_graph_edges, train_solution_edges, train_terminals, test_graph_edges, test_solution_edges, test_terminals = load_data(folder_name)
print("20 nodes training size:", len(train_graph_edges))
print("20 nodes testing size:", len(test_graph_edges))

#folder_name = "/Users/abureyanahmed/GNN/TSP_Steiner/Steinertree-master/steiner_dataset50/"
folder_name = "./exp_ER_GNN_Steiner_TSP_50/"
train_graph_edges50, train_solution_edges50, train_terminals50, test_graph_edges50, test_solution_edges50, test_terminals50 = load_data(folder_name)
print("50 nodes training size:", len(train_graph_edges50))
print("50 nodes testing size:", len(test_graph_edges50))
#train_graph_edges, train_solution_edges, train_terminals, test_graph_edges, test_solution_edges, test_terminals = train_graph_edges50, train_solution_edges50, train_terminals50, test_graph_edges50, test_solution_edges50, test_terminals50
#test_graph_edges, test_solution_edges, test_terminals = test_graph_edges50, test_solution_edges50, test_terminals50

train_graph_edges, train_solution_edges, train_terminals, test_graph_edges, test_solution_edges, test_terminals = train_graph_edges + train_graph_edges50, train_solution_edges + train_solution_edges50, train_terminals + train_terminals50, test_graph_edges + test_graph_edges50, test_solution_edges + test_solution_edges50, test_terminals + test_terminals50
'''

folder_name = "./exp_all_GNN_spanner/"
train_graph_edges, train_solution_edges, train_terminals, test_graph_edges, test_solution_edges, test_terminals = load_data(folder_name)
#print("test_graph_edges[0]", test_graph_edges[0])
print("20 nodes training size:", len(train_graph_edges))
print("20 nodes testing size:", len(test_graph_edges))

def plot_steiner_dist(total_graph_edges, total_solution_edges, total_terminals):
 steiner_arr = []
 for i, sol_edges in enumerate(total_solution_edges):
  sol_nodes = set()
  for u, v, w in sol_edges:
   sol_nodes.add(u)
   sol_nodes.add(v)
  steiner_arr.append(len(sol_nodes)-len(total_terminals[i][0]))
 print(steiner_arr)
 from matplotlib import pyplot
 pyplot.hist(steiner_arr)
 pyplot.title("Steiner nodes")
 pyplot.savefig("steiner_node_distribution.png", bbox_inches='tight')
 quit()

#total_graph_edges, total_solution_edges, total_terminals = train_graph_edges+test_graph_edges, train_solution_edges+test_solution_edges, train_terminals+test_terminals
#plot_steiner_dist(total_graph_edges, total_solution_edges, total_terminals)

# generate vector
'''
train_features, train_labels, n_inputs, n_outputs = generate_dataset(train_graph_edges, train_terminals, train_solution_edges)
test_features, test_labels, _, _ = generate_dataset(test_graph_edges, test_terminals, test_solution_edges)
'''
#'''
train_features, train_labels, n_inputs, n_outputs = generate_dataset(train_graph_edges, train_terminals, train_solution_edges, labels = 'edges')
test_features, test_labels, _, _ = generate_dataset(test_graph_edges, test_terminals, test_solution_edges, labels = 'edges')
#'''

# train
activation_function = 'relu'
'''
units_in_layer1 = 6
units_in_layer2 = 6
'''
'''
units_in_layer1 = 100
units_in_layer2 = 100
'''
#'''
units_in_layer1 = 300
units_in_layer2 = 300
#'''
output_activation_function = 'sigmoid'
loss_string = 'binary_crossentropy'
relu_neural_network = tensorflow.keras.Sequential()
relu_neural_network.add(tensorflow.keras.layers.Dense(units_in_layer1, activation=activation_function, input_shape=(n_inputs,)))
relu_neural_network.add(tensorflow.keras.layers.Dense(units_in_layer2, activation=activation_function))
#'''
relu_neural_network.add(tensorflow.keras.layers.Dense(units_in_layer2, activation=activation_function))
relu_neural_network.add(tensorflow.keras.layers.Dense(units_in_layer2, activation=activation_function))
#'''
relu_neural_network.add(tensorflow.keras.layers.Dense(n_outputs, activation=output_activation_function))
relu_neural_network.compile(loss=loss_string, optimizer=tensorflow.keras.optimizers.Adam(0.001))
#epochs = 10
epochs = 20
batch_size = 10
relu_neural_network.fit(train_features, train_labels, epochs=epochs, batch_size=batch_size)

# test
# check that error levels are acceptable
prediction = relu_neural_network.predict(test_features)
relu_accuracy = binary_accuracy(prediction, test_labels)
accuracy_format = "{1:.1%} accuracy for {0} on Steiner dataset".format
print()
print(accuracy_format("relu_neural_network", relu_accuracy))
precision_recall(prediction, test_labels)

def generate_solution_graphs(prediction, test_graph_edges, test_solution_edges, test_terminals):
 n_inputs, n_outputs, max_edge_index, edge_to_id, id_to_edge = edge_to_number()
 connected_graphs = 0
 disconnected_graphs = 0
 disconnected_opt_solutions = 0
 rat_arr = []
 for j, pred in enumerate(prediction):
  sorted_ind = np.argsort(pred)
  #print(pred)
  #print(sorted_ind)
  pred = np.round(pred)
  opt = len(test_solution_edges[j])
  G = nx.Graph()
  for e in test_graph_edges[j]:
   u, v, w = e
   G.add_edge(u, v)
  V = []
  for i, val in enumerate(pred):
   if val==1:
    V.append(i)
    #G.add_edge(*id_to_edge[i])
  for terminal in test_terminals[j][0]:
   if terminal not in V:
    V.append(terminal)
  S = G.subgraph(V)
  #'''
  is_disconnected = False
  if not nx.is_connected(S):
   is_disconnected = True
  k = len(sorted_ind)-1
  while not nx.is_connected(S):
   u = sorted_ind[k]
   if u not in V:
    V.append(u)
    S = G.subgraph(V)
   k -= 1
  if is_disconnected:
   mst=nx.minimum_spanning_edges(S)
   apprx = len(list(mst))
   if opt==apprx:
    disconnected_opt_solutions += 1
  #'''
  if nx.is_connected(S):
   connected_graphs += 1
   mst=nx.minimum_spanning_edges(S)
   apprx = len(list(mst))
   rat_arr.append(apprx/opt)
  else:
   disconnected_graphs += 1
 print("No. of connected graphs:", connected_graphs)
 total_rat = 0
 for rat in rat_arr:
  total_rat += rat
 avg_rat = total_rat/connected_graphs
 print("average ratio:", avg_rat)
 print("No. of disconnected graphs:", disconnected_graphs)
 print("No. of disconnected graphs become optimal after connecting:", disconnected_opt_solutions)

def find_max_weight(G):
  mx = 0
  for e in G.edges():
    u, v = e
    wgt = G.get_edge_data(u, v, 'weight')['weight']
    if mx<wgt:
      mx = wgt
  return mx

def all_pairs_from_subset(s):
        s = list(s)
        all_pairs = []
        for i in range(len(s)):
                for j in range(i+1, len(s)):
                        p = []
                        p.append(s[i])
                        p.append(s[j])
                        all_pairs.append(p)
        return all_pairs

def check_stretch(spanner_sp, graph_sp, param):
  if spanner_sp<=(graph_sp+param):
    return True
  return False

def verify_spanner_with_checker(G_S, G, all_pairs, check_stretch, param):
        for i in range(len(all_pairs)):
                if not (all_pairs[i][0] in G_S.nodes() and all_pairs[i][1] in G_S.nodes()):
                        return False
                if not nx.has_path(G_S, all_pairs[i][0], all_pairs[i][1]):
                        return False
                sp = nx.shortest_path_length(G, all_pairs[i][0], all_pairs[i][1], 'weight')
                #if not check_stretch(nx.dijkstra_path_length(G_S, all_pairs[i][0], all_pairs[i][1]), sp, param):
                if not check_stretch(nx.shortest_path_length(G_S, all_pairs[i][0], all_pairs[i][1], 'weight'), sp, param):
                        return False
        return True

def generate_solution_graphs_from_edge_labels(prediction, test_graph_edges, test_solution_edges, test_terminals):
 n_inputs, n_outputs, max_edge_index, edge_to_id, id_to_edge = edge_to_number()
 connected_graphs = 0
 disconnected_graphs = 0
 rat_arr = []
 for j, pred in enumerate(prediction):
  #print("Graph", j)
  #print("edge_to_id", edge_to_id)
  #print("id_to_edge", id_to_edge)
  sorted_ind = np.argsort(pred)
  #print(pred)
  #print(sorted_ind)
  pred = np.round(pred)
  #opt = len(test_solution_edges[j])
  G = nx.Graph()
  for u, v, w in test_graph_edges[j]:
   G.add_edge(u, v, weight=w)
  opt = 0
  for u, v, w in test_solution_edges[j]:
   opt += w
  #pos = nx.spring_layout(G)
  #nx.draw(G, pos=pos, with_labels=True)
  #plt.show()
  S = nx.Graph()
  for terminal in test_terminals[j][0]:
   S.add_node(terminal)
  for i, val in enumerate(pred):
   if val==1:
    u, v = id_to_edge[i]
    if not G.has_edge(u, v):
     continue
    S.add_edge(u, v, weight=G[u][v]['weight'])
    '''
    try:
     nx.find_cycle(S, orientation="original")
     S.remove_edge(u, v)
    except:
     pass
    '''
  st = test_terminals[j][0]
  max_weight = find_max_weight(G)
  '''
  k = len(sorted_ind)-1
  #num_prev_edges = len(list(S.edges()))
  #while not nx.is_connected(S):
  while not verify_spanner_with_checker(S, G, all_pairs_from_subset(st), check_stretch, 2*max_weight):
   #print(S.edges())
   #if num_prev_edges<len(list(S.edges())):
    #pos = nx.spring_layout(S)
    #nx.draw(S, pos=pos, with_labels=True)
    #plt.show()
   #num_prev_edges = len(list(S.edges()))
   if k<0:break
   e = sorted_ind[k]
   u, v = id_to_edge[e]
   #if u==0:
   # print("found an edge from 0 (", u, v, ")")
   if not G.has_edge(u, v):
    k -= 1
    continue
   #if u==0:
   # print("found an edge from 0 (", u, v, ")")
   S.add_edge(u, v, weight=G[u][v]['weight'])
   k -= 1
  '''
  #if nx.is_connected(S):
  if verify_spanner_with_checker(S, G, all_pairs_from_subset(st), check_stretch, 2*max_weight):
   #print(test_terminals[j][0])
   #print(test_solution_edges[j])
   #print(S.edges())
   connected_graphs += 1
   #apprx = len(list(S.edges()))
   apprx = 0
   for u, v in S.edges():
    apprx += G[u][v]['weight']
   if apprx/opt<1:
    print(test_graph_edges[j])
    print(test_terminals[j][0])
    print(test_solution_edges[j])
    print(S.edges())
    quit()
   rat_arr.append(apprx/opt)
  else:
   disconnected_graphs += 1
  #break
 print("No. of valid spanners:", connected_graphs)
 total_rat = 0
 for rat in rat_arr:
  total_rat += rat
 if not connected_graphs==0:
  avg_rat = total_rat/connected_graphs
 else:
  avg_rat = 1
 #if avg_rat<1:
 # avg_rat = 1
 print("average ratio:", avg_rat)
 print("No. of invalid spanners:", disconnected_graphs)

#generate_solution_graphs(prediction, test_graph_edges, test_solution_edges, test_terminals)
generate_solution_graphs_from_edge_labels(prediction, test_graph_edges, test_solution_edges, test_terminals)





