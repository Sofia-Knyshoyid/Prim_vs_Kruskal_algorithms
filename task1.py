import random
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import tree
from itertools import combinations, groupby
import time
import numpy as np
from tqdm import tqdm

def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               draw: bool = False) -> list[tuple[int, int]]:
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    weights=[]
    edges = combinations(range(num_of_nodes), 2)
    G = nx.Graph()
    G.add_nodes_from(range(num_of_nodes))
    
    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)
                
    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(0,10)
        weights.append(w['weight'])

    if draw: 
        plt.figure(figsize=(10,6))
        nx.draw(G, node_color='lightblue', 
            with_labels=True, 
            node_size=500)
    
    return G

def turn_into_matrix(number, compl):
    list_of_edges=list(gnp_random_connected_graph(number,compl).edges.data('weight'))
    adjacencyMatrix = [[0 for column in range(number)] for row in range(number)]
    for i,j,k in list_of_edges:
        adjacencyMatrix[i][j]=k
        adjacencyMatrix[j][i]=k
    return adjacencyMatrix

def prim(adjmatrix):
    selected_node=[False]*len(adjmatrix)
    total=0
    length=0
    selected_node[0]=True
    while length<len(adjmatrix)-1:
        min=999999999
        u=0
        v=0
        for i in range(len(adjmatrix)):
            if selected_node[i]:
                for j in range(len(adjmatrix)):
                    if ((not selected_node[j]) and adjmatrix[i][j]):  
                        if min > adjmatrix[i][j]:
                            min = adjmatrix[i][j]
                            u = i
                            v = j
        total+=adjmatrix[u][v]
        selected_node[v] = True
        length+=1
    return total

def counting_time(nodes, NUM_OF_ITERATIONS):
    time_taken = 0
    for i in tqdm(range(NUM_OF_ITERATIONS)):
    
        G = turn_into_matrix(nodes,0.1)
    
        start = time.time()
        prim(G)
        end = time.time()
    
        time_taken += end - start
    return time_taken

def kruskal_alg(edges_ls:list):
    """
    the Kruskal algorithm
    """
 
    # since we take G = gnp_random_connected_graph(..,..,draw=True),
    # take edges_ls as G.edges(data=True)
    # edges_ls looks like: [(11, 12, {'weight': 7}), ...]
    # step 1: sorting the edges by weight
    sorted_edg_ls = sorted(edges_ls, key=lambda x: x[2]['weight'])
 
    # creating containers of joined nodes,
    # isolated, independent nodes groups
    # and final result edges of Kruskal algorithm
    joined, group, result_edges = set(), dict(), list()
 
    one_group = True # assume everything is in one connected group as nodes are connected
    for edge in sorted_edg_ls:
        first_node = edge[0]
        second_node = edge[1]
        # join all the nodes to some joined group; in result, we get several independent structures of connected nodes
        if first_node not in joined or second_node not in joined:  # if both are already connected, there is a risk of creating a cycle
            if first_node not in joined and second_node not in joined:
                group[first_node] = group[second_node] = [first_node, second_node]  # add both nodes to group dictionary
            else:
                if first_node not in group.keys():             # first node is not in a group
                    group[second_node].append(first_node)        # put it in the group with the second node
                    group[first_node] = group[second_node]
                else:
                    group[first_node].append(second_node)     # second node is not in the group, put it in the group with node 1
                    group[second_node] = group[first_node]
            # since we have connected two nodes appropriately, add this pair to the resulting graph
            result_edges.append(edge)
            joined.add(first_node)
            joined.add(second_node)
 
    # now join the groups of nodes we got
    possible_connection_ls = []
    for edge in sorted_edg_ls:
        first_node = edge[0]
        second_node = edge[1]
        if second_node not in group[first_node]:     # nodes are in different groups
            one_group = False
            possible_connection_ls.append(edge)
    if one_group is False:
        final_join = sorted(possible_connection_ls, key=lambda x: x[2]['weight'])
        result_edges.append(final_join[0])
    min_sum = 0
    for elem in result_edges: # count sum of the resulting graph
        min_sum += elem[2]['weight']
    return min_sum

def counting_time_2(nodes,NUM_OF_ITERATIONS):
    time_taken_2=0
    for i in tqdm(range(NUM_OF_ITERATIONS)):
    
        G=gnp_random_connected_graph(nodes,0.1)
    
        start = time.time()
        kruskal_alg(G.edges(data=True))
        end = time.time()
    
        time_taken_2 += end - start
    return time_taken_2

def draw_graph(dct1,dct2):
    row_1=list(dct1.keys())
    row_2=list(dct1.values())
    plt.plot(row_1, row_2, 'r', marker='o', label="Алгоритм Прима")
    row_1=list(dct2.keys())
    row_2=list(dct2.values())
    plt.plot(row_1, row_2, 'b', marker='o', label='Алгоритм Краскала')
    plt.title('Часова ефективність алгоритмів')
    plt.xlabel('Розмірність графу')
    plt.ylabel('Час роботи')
    plt.legend()
 
    plt.show()

if __name__=="__main__":
    dict_of_results={}
    dict_of_results[10]=counting_time(10,1000)
    dict_of_results[20]=counting_time(20,1000)
    dict_of_results[50]=counting_time(50,1000)
    dict_of_results[100]=counting_time(100,1000)
    dict_of_results[200]=counting_time(200,1000)
    dict_of_results[250]=counting_time(250,1000)
    print(dict_of_results)
    second={}
    second[10]=counting_time_2(10,1000)
    second[20]=counting_time_2(20,1000)
    second[50]=counting_time_2(50,1000)
    second[100]=counting_time_2(100,1000)
    second[200]=counting_time_2(200,1000)
    second[250]=counting_time_2(250,1000)

    print(second)
    draw_graph(dict_of_results,second)
