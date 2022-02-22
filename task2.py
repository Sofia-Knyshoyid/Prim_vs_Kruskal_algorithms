import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

def group_elem_num(group):
        """
        returns the number of elements in a group
        """
        # group looks like {'type1':num1, 'type2':num2}
        sum = 0
        for key in group:
            sum += group[key]
        return sum

class Node:
    def __init__(self, gini, total_elem_num, class_num_ls, chosen_class):
        self.gini = gini
        self.total_elem_num = total_elem_num
        self.class_num_ls = class_num_ls
        self.chosen_class = chosen_class
        self.node_split_idx = 0
        self.node_thr = 0
        self.left = None
        self.right = None
        
class MyDecisionTreeClassifier:
    
    def __init__(self, max_depth):
        self.max_depth = max_depth
        
    # we split data, and get two groups of data
    def gini(self, groups, classes, which_gini='total', list_input=False):
        '''
        A Gini score gives an idea of how good a split is by how mixed the
        classes are in the two groups created by the split.
        
        A perfect separation results in a Gini score of 0,
        whereas the worst case split that results in 50/50
        classes in each group result in a Gini score of 0.5
        (for a 2 class problem).
        '''
        # for now, assume that groups look like [{'type1':num1, 'type2':num2},{'type1':num1, 'type2':num2}]
        # and classes look like ['type1', 'type2', 'type3']
        if list_input is True: # convert [0,0,0,1,1,2,3,3] --> {0:3, 1:2, 2:1, 3:2} for both groups
            for group_idx in len(groups):
                new_dict = {}
                for elem in groups[group_idx]:
                    if elem not in new_dict:
                        new_dict[elem] = 1
                    else:
                        new_dict[elem] += 1
                groups[group_idx] = new_dict
            

        if which_gini == 'group1' or which_gini=='total':
            group_1 = groups[0]
            gr_1_elem_num = group_elem_num(group_1)
            group_1_gini = 1
            for type in group_1:
                group_1_gini -= (group_1[type]/gr_1_elem_num)**2
            print('gini 1 is:')
            print(group_1_gini)
            print()

        if which_gini == 'group2' or which_gini=='total':
            group_2 = groups[1]
            gr_2_elem_num = group_elem_num(group_2)
            group_2_gini = 1
            for type in group_2:
                group_2_gini -= (group_2[type]/gr_2_elem_num)**2
            print('gini 2 is:')
            print(group_2_gini)
            print()
        
        if which_gini == 'group1':
            return group_1_gini

        elif which_gini == 'group2':
            return group_1_gini

        elif which_gini=='total':
            total_el_num = gr_1_elem_num + gr_2_elem_num
            final_gini = (gr_1_elem_num/total_el_num)*group_1_gini + (gr_2_elem_num/total_el_num)*group_2_gini
            return final_gini


    def split_data(self, X, y):
        """
        Function defines the best split
        returns the spli index and a threshold value
        """
        chosen_thr, chosen_feature = None, None
        start_gini = 1 # set it to the high value only to reduce it eventually
        # for idx, row in df.iterrows():
        transposed_dataframe = X.T
        for idx in range (len(transposed_dataframe)):
            # chosen feature = transposed_dataframe[idx]
            # from [[5.1 3.5 1.4 0.2]
               #     [4.9 3.  1.4 0.2]
                 #   [4.7 3.2 1.3 0.2]
                  #  [4.6 3.1 1.5 0.2]
                  #  [5.  3.6 1.4 0.2]]
            # we take (0:) [5.1 4.9 4.7 4.6 5. ] and so on

            # the principle of this part is searching the best gini value
            # the logic is quite simple: we set gini to max of 1,
            # if we can find lower gini - set it as the desirable output;
            # repeat those steps, until all variations of gini are checked 
            used_features = set()
            used = []
            changed_ls = [x for x in transposed_dataframe[idx] if x not in used and used.append(x)]
            # set as high as 1 to change that later
            gini_minim = 1
            for datum in changed_ls:  # checking data one by one in transposed frame
                if datum in used_features:
                    continue # we have already discovered this one, continue iterating
                bool_dataframe = X[:,idx] < datum # divide our dataset into 2 groups;
                part_1 = X[bool_dataframe]
                part_2 = X[~bool_dataframe]
                dict_1 = {}
                dict_2 = {}
                for elem in part_1:
                    if elem[idx] in dict_1:
                        dict_1[elem[idx]] += 1
                    else:
                        dict_1[elem[idx]] = 1
                for elem in part_2:
                    if elem[idx] in dict_2:
                        dict_2[elem[idx]] += 1
                    else:
                        dict_2[elem[idx]] = 1
                total_ls = [dict_1, dict_2]
                curr_gini = self.gini(total_ls, y)
                if curr_gini <= gini_minim:
                    gini_minim, threshold = curr_gini, datum # change our minimal value to better of a current gini,
                                                                # remember threshold to return it later
                used_features.add(datum)

            if curr_gini <= start_gini:
                start_gini = curr_gini # we replace the start gini with the best one evntually
                chosen_feature = idx
                chosen_thr = threshold
        return chosen_feature, chosen_thr, start_gini

          
    def build_tree(self, X, y, depth=0):
            """
            function builds a decision tree
            it uses previously mentioned the split_data function to perform
            creates a root node
            uses the depth parameter from the arguments provided by the user
            """

            # here we represent classes by their index:
            # class_1 - 0, class_2 - 1 and so on
            # therefore, we use fitting here list type
            num_ls = [0]*class_num
            y = list(y) # ensuring the list type; this way, code is valid for other types of arrays
            for i in range(class_num): #  we define how many elements are in each class
                num_ls[i] = y.count(i)

            # print(num_ls)
            max_use_class = num_ls.index(max(num_ls))
            # print(max_use_class)

            chosen_vertex = Node(gini=self.gini(y), total_elem_num=len(y), class_num_ls=num_ls, chosen_class=max_use_class)
            # now we have to use our previously implemented split function
            # the max depth for splitting was taken as an argument from the user
            y = np.array(y)
            # np array will be user further for y
            if depth >= self.max_depth:
                print('The argument passed exceeds the lowest level of the tree')
                print('If you want to build a tree further, please, change the value')
            else:  # The level passed to function is lower than the maximum. We can still build 1+ levels to the tree
                        split_result = list(self._best_split(X, y)) # split with previously implemented function
                        split_index, split_threshold = split_result[0], split_result[1]
                        if True not in list(map(lambda x:isinstance(x, None))): # check if split got us valid data
                            chosen_data_set = X[:,split_index]
                            left_part = chosen_data_set < split_threshold # divide dataframe by bool
                            iter_pair = [X, y]
                            left_data_y_pair = tuple(map(lambda x: x[left_part], iter_pair))
                            # note: y is assumed to be np array, so it can be sorted in same way
                            right_data_y_pair = tuple(map(lambda x: x[~left_part], iter_pair))
                            chosen_vertex.node_split_idx = split_index
                            chosen_vertex.node_thr = split_threshold
                            # here we use recursive call to build the left and right subtrees
                            # the max depth will be eventually reached, so building will stop
                            current_depth = depth + 1
                            # get to the next lower level
                            chosen_vertex.left = self.build_tree(left_data_y_pair[0], left_data_y_pair[1], current_depth)
                            chosen_vertex.right = self.build_tree(right_data_y_pair[0], right_data_y_pair[1], current_depth)
                    # if the data is invalid we can't change current node (add subtrees etc)
                    # perhaps, it can be a leaf; in this case, no adjustments are made, the node is returned as it is
                    # recursion in previous call will ensure building the other branches until the end of process
            return chosen_vertex
