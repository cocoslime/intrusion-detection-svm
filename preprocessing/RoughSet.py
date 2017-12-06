import pandas as pd
import numpy as np


class RoughSet():
    def getRelation(self,series):
        result = [];
        curr = [];
        series = series.sort_values();
        curr.append(series.index[0])
        for i in range(1, len(series.values)):
            if (series.values[i-1]== series.values[i]):
                curr.append(series.index[i])
            else:
                result.append(curr)
                curr = [series.index[i]]
        result.append(curr)
        return result
    def getIND(self,P):
        # if (len(P) == 1) : return self.relation[P[0]];
        # zero_delete = self.getIND(np.delete(P, 0))
        IND = self.relation[P[0]];
        for idx,p_i in enumerate(P):
            if idx == 0 : continue
            from calculate.set import innerIntersect
            IND = innerIntersect(IND, self.relation[p_i])
            # print(idx,p_i)
        return IND
    def getPOS(self,P):
        IND_Q = self.relation['type'];
        IND_P = self.getIND(P);
        result = set()
        for i in range(0, len(IND_Q)):
            for j in range(0, len(IND_P)):
                if set(IND_P[j]).issubset(IND_Q[i]):
                    result = result.union(set(IND_P[j]))

        return result;
    # get degree of dependency
    def getDOD(self,P):
        pos = self.getPOS(P)
        return len(pos) / self.U_size;
    def getReducts(self,data):
        columns = data.columns;
        self.U_size = len(data);
        self.relation = {};
        data = data.reset_index()
        for col in columns:
            print("make relation : ", col)
            self.relation[col] = self.getRelation(data[col])
        # column_array = [np.delete(columns, len(columns)-1).tolist()]

        features = np.delete(columns, len(columns)-1).tolist()
        if self.getDOD(features) != 1.0 :
            print("Error")
            return
        column_array = []
        import itertools
        for i in range(1, len(features) - 1):
            print("combination : ", i)
            for sub_tuple in itertools.combinations(features, i):
                subset = set(sub_tuple)
                issuperset = False;
                for column in column_array:
                    if subset.issuperset(column):
                        issuperset = True;
                        break;
                if issuperset:
                    continue;
                if self.getDOD(list(sub_tuple)) == 1.0:
                    column_array.append(list(sub_tuple))

        # for i in range(0, len(columns) - 1):
        #     print("Level ",i);
        #     new_column_array = []
        #     for j in range(0, len(column_array)):
        #         columns_elements = column_array[j];
        #         new_column_array.append(columns_elements);
        #         # print ("remove %s in %d"%(columns[i], j))
        #         remove_i_columns = list(columns_elements);
        #         remove_i_columns.remove(columns[i]);
        #         if self.getDOD(remove_i_columns) == 1.0 :
        #             new_column_array.append(remove_i_columns)
        #     column_array = new_column_array;
        reducts = self.minimize(column_array)
        return reducts
    def getFeatureWeights(self,data):
        reducts = self.getReducts(data)
        weights = dict.fromkeys(np.delete(data.columns, len(data.columns) - 1))
        for i in range(0, len(data.columns)-1):
            for red in reducts:
                if weights[i] == None:
                    weights[i] = 0
                weights[i] = weights[i] + 1/reducts;
        norm_weights = [float(i) * 100/max(weights) for i in weights]
        print(norm_weights)
        return norm_weights;

    def minimize(self, array):
        sorted_array = sorted(array, key=len)
        print(sorted_array)
        return sorted_array