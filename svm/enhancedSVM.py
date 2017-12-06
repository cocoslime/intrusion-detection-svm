class EnhancedSVM():
    def train(self,data):
        from preprocessing.RoughSet import RoughSet
        rough_set_data = data.loc[0:1000,:];
        weights = RoughSet().getFeatureWeights(rough_set_data);
        return
    def test(self,test_X):
        return
