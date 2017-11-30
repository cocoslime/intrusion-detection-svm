import pandas as pd
def filter(data, intrusion_ratio):
    print("whole : ", data.shape[0])
    intrusion_data = data[data.type != 0]
    normal_data = data[data.type == 0]
    print("normal : ", normal_data.shape[0])

    intrusion_num = normal_data.shape[0] * intrusion_ratio
    print("intrusion : ", intrusion_num)
    intrusion_data = intrusion_data.sample(int(intrusion_num))

    data = pd.concat([normal_data,intrusion_data])
    data = data.sample(frac=1).reset_index(drop=True)
    return data

class myLabelEncoder():
    def __init__(self):
        from collections import defaultdict
        from sklearn.preprocessing import LabelEncoder
        self.d = defaultdict(LabelEncoder)
    def encode(self, data):
        # 0 is normal, 1 is intrusion
        data.loc[data.type != 'normal.', 'type'] = 1;
        data.loc[data.type == 'normal.', 'type'] = 0;

        fit = data.apply(lambda x : self.d[x.name].fit_transform(x))
        return fit