import networkx as nx
from input_functions import *

def bellman_ford(G_cspp, MAX_WEIGHT, MAX_COST, s, d):
 RC_EPS = 1.0e-6
 #RC_EPS = .5
 #print('MAX_WEIGHT, MAX_COST, s, d:', MAX_WEIGHT, MAX_COST, s, d)
 my_inf = 10000000
 #min cost to path with t hops s->v : d[v, t]
 dis = []
 par = []
 # vertex set 0, 2, ..., n-1
 n = len(G_cspp.nodes())
 for i in range(0, n):
  dis.append([])
  par.append([])
  #for j in range(0, 10*n):
  for j in range(0, 11*n):
   if i==s and j==0:
    dis[i].append(0)
    par[i].append(-1)
   else:
    dis[i].append(my_inf)
    par[i].append(-1)
 #print(dis)
 m = len(G_cspp.edges())
 #for i=1 to n-1
 for i in range(0, n-1):
  #for t=1 to 10n
  #for t in range(1, 10*n):
  for t in range(1, 11*n):
   #for (u,v) in E
   for u, v in G_cspp.edges():
    #d[v, t] = min{d[v, t], d[u, t-1]+w(w,v)}
    if dis[v][t] > dis[u][t-1] + G_cspp[u][v]['cost']:
     par[v][t] = u
     dis[v][t] = dis[u][t-1] + G_cspp[u][v]['cost']
    if dis[u][t] > dis[v][t-1] + G_cspp[u][v]['cost']:
     par[u][t] = v
     dis[u][t] = dis[v][t-1] + G_cspp[u][v]['cost']
 min_i = -1
 cost = -1
 weight = -1
 path_graph = nx.Graph()
 #for i in range(10*n):
 for i in range(11*n):
  if dis[d][i]<=MAX_COST:
  #if dis[d][i]<MAX_COST:
  #if dis[d][i]<MAX_COST+RC_EPS:
  #if (dis[d][i]<MAX_COST) or (dis[d][i]-MAX_COST<RC_EPS) or (MAX_COST-dis[d][i]<RC_EPS):
   #print('dis[d][i]<=MAX_COST')
   curr_cost = dis[d][i]
   curr_weight = 0
   curr_i = i
   curr_node = d
   curr_par = par[d][curr_i]
   temp = nx.Graph()
   while curr_par!=-1:
    temp.add_edge(curr_par, curr_node, weight=G_cspp[curr_par][curr_node]['weight'], cost=G_cspp[curr_par][curr_node]['cost'])
    #print(curr_par, curr_node, G_cspp[curr_par][curr_node]['weight'], G_cspp[curr_par][curr_node]['cost'])
    curr_weight += G_cspp[curr_par][curr_node]['weight']
    curr_i = curr_i - 1
    curr_node = curr_par
    curr_par = par[curr_node][curr_i]
   #print('curr_weight:', curr_weight)
   if curr_weight<=MAX_WEIGHT:
    #if (min_i == -1) or (weight>curr_weight):
    if (min_i == -1) or (cost>curr_cost):
     min_i = i
     cost = curr_cost
     weight = curr_weight
     path_graph = temp
 return (path_graph, cost, weight)

#G, subset_arr = build_networkx_graph('erdos_renyi_sm2/graph_1.txt')
#print('nodes:', G.nodes())
#print('edges:', G.edges())
#for u,v in G.edges():
# print(u, v)

#G = nx.Graph()
#G.add_edge(0, 1, weight=2, cost=1)
#G.add_edge(1, 2, weight=2, cost=1)
#G.add_edge(1, 6, weight=1, cost=2)
#G.add_edge(2, 3, weight=2, cost=1)
#G.add_edge(6, 7, weight=1, cost=2)
#G.add_edge(3, 4, weight=2, cost=1)
#G.add_edge(7, 8, weight=1, cost=2)
#G.add_edge(4, 5, weight=2, cost=1)
#G.add_edge(8, 5, weight=1, cost=2)
#print(bellman_ford(G, 100, 20, 1, 5))


