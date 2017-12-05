import pandas as pd
import numpy as np

class RoughSet():
    def getRelation(self,data,q):
        sorted = data.sort([q]);
        #for index in range(0, sorted):
        return
    def getIND(self,data,P):
        return
    def getPOS(self,data,P):

        return
    # get degree of dependency
    def getDOD(self,data,P):
        pos = self.getPOS(data,P)
        return len(pos) / self.U_size;
    def getReducts(self,data):
        columns = data.columns
        self.U_size = len(columns);
        self.relation = {};
        data = data.reset_index()
        # data['index_col'] = data.index;
        for col in columns:
            self.relation[col] = self.getRelation(data,col)
        # data.drop('index_col', 1, inplace=True)
        remove_col = []
        while(True):
            for idx, col in enumerate(columns):
                # if degree of dependency of (columns-col) == 1
                # remove col
                if self.getDOD(data, np.delete(columns, idx)) == 1.0 :
                    remove_col.append(idx)
        return
    def getFeatureWeights(self,data):
        reducts = self.getReducts(data)