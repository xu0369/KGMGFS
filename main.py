import pandas as pd
import numpy as np
import warnings
import copy
import time
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
warnings.filterwarnings('ignore')


from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance

class KGMGFS:
    def __init__(self,data):
        self.data = data
        self.data_test=data
        self.label_name = None
        self.selected_data = []
        self.data_backup = None
        self.class_name = list(self.data)[-1]
        self.label=data.iloc[:,-1]
        self.attr_names = list(self.data)[0:-1]
        self.attr_num = len(list(self.data))-1
        self.sample_num = len(self.data)
        self.attr_list = [i for i in range(self.attr_num)]
        self.attr_sort = []
        self.origin_fuzzy_list = [[] for i in range(self.attr_num)]
        self.fuzzy_list = [[] for i in range(self.attr_num)]
        self.res = [[] for i in range(30)]
        self.judge_attr = [[] for i in range(self.attr_num)]
        self.D_fuzzy_mat=[]
        self.FNGMI=[None for i in range(self.attr_num)]
        self.FNGE=[None for i in range(self.attr_num)]
        self.Max_MAA=[]
        self.max=[None for i in range(self.attr_num)]
        self.REL_S=None
        self.param_L=1

    def data_deal(self):
        nominal_list = []
        encode=LabelEncoder()
        for i,name in enumerate(self.attr_names):
            # sim_mat=np.zeros((self.sample_num,self.sample_num))
            if(np.issubdtype(self.data[name],np.number)==True):
                minmax_deal = MinMaxScaler()
                self.data[[name]] = minmax_deal.fit_transform(self.data[[name]])
                raids=np.std(self.data[name],ddof=1)

                vector = self.data.values[:,i].reshape(1,self.sample_num)
                dis_mat = abs(vector.T-vector)

                sim_mat=np.where(dis_mat<=raids/self.param_L,1,0)
                self.origin_fuzzy_list[i] = np.float64(sim_mat)

            else:
                nominal_list.append(name)
                vector = self.data.values[:,i].reshape(1,self.sample_num)
                self.origin_fuzzy_list[i] = (vector==vector.T).astype(int)
                self.data.iloc[:,i]=encode.fit_transform(self.data.iloc[:,i])

        if(len(nominal_list)!=0):
            attr_encode = OneHotEncoder()
            x_encode = attr_encode.fit_transform(self.data[nominal_list]).toarray()
            self.data_backup = pd.concat([self.data,pd.DataFrame(x_encode)],axis=1)
            self.data_backup.drop(nominal_list,axis=1,inplace=True)
            self.data_backup.drop([self.class_name],axis=1,inplace=True)
        else:
            self.data_backup = copy.deepcopy(self.data)
            self.data_backup.drop([self.class_name],axis=1,inplace=True)

        label_encode = OneHotEncoder()
        label_dis = label_encode.fit_transform(self.data[[self.class_name]]).toarray()
        self.data = pd.concat([self.data, pd.DataFrame(label_dis, columns=label_encode.get_feature_names_out(['Class']))],axis=1)
        self.data.drop(self.class_name, axis=1, inplace=True)
        self.label_name = label_encode.get_feature_names_out(['Class'])
        print(self.data)

    def cal_fuzzy_D(self):
        D_mat = distance.cdist(self.data[self.label_name].values,self.data[self.label_name].values,'euclidean')
        self.D_fuzzy_mat = np.exp(np.array(-D_mat/((2*np.percentile(D_mat,20,axis=1)+0.001).reshape(self.sample_num,1)),dtype=np.float64))
        self.D_entropy = np.sum(1-np.sum(self.D_fuzzy_mat,axis=1)/self.sample_num)/self.sample_num

    def cal_fuzzy_list(self):
        for i,name in enumerate(self.attr_names):
            self.judge_attr[i] = np.std(self.data[name]) /self.param_L
            replace_mat = self.origin_fuzzy_list[i]
            self.fuzzy_list[i] = np.where(replace_mat < 1-self.judge_attr[i],0,replace_mat)

    def first_reduction(self):
        for i in self.attr_list:
            self.FNGMI[i]=np.sum(np.sum(np.minimum(1-self.fuzzy_list[i],1-self.D_fuzzy_mat))/self.sample_num)/self.sample_num

        self.attr_list = np.argsort(self.FNGMI)[::-1].tolist()
        print(self.attr_list)

    def attr_rduction(self):
        self.data_deal()
        param_L = 0.5
        step = 0.1

        for item in range(1):
            self.cal_fuzzy_list()
            self.cal_fuzzy_D()
            self.first_reduction()
            Red=self.fuzzy_list[self.attr_list[0]]
            self.attr_sort.append(self.attr_list[0])
            self.attr_list.remove(self.attr_list[0])
            attr_list_red=self.attr_list.copy()
            start=1
            for i in attr_list_red:
                print(i+1)
                Red_a=np.minimum(self.fuzzy_list[i],Red)
                Rel_red=np.sum(np.sum(np.minimum(1-Red,1-self.D_fuzzy_mat))/self.sample_num)/self.sample_num
                Rel_red_a=np.sum(np.sum(np.minimum(1-Red_a,1-self.D_fuzzy_mat))/self.sample_num)/self.sample_num
                valve=Rel_red_a-Rel_red
                print(valve)
                if(valve>0):
                    self.attr_sort.append(i)
                    Red=Red_a
                self.attr_list.remove(i)
                start+=1
                # if(start==100):
                if(start==50):
                    break

            print(self.attr_sort,len(self.attr_sort))
            self.attr_sort.clear()
            param_L+=step

    def run(self):
        start_time = time.time()
        self.attr_rduction()
        end_time = time.time()
        elapsed_time = end_time-start_time
        print(f"{elapsed_time:.4f}")



df= pd.DataFrame({
    'c1': [0.25,1.3,0,1.5,2.6,3.6],
    'c2': [41, 25, 38, 72, 67, 62],
    'c3': [1, 1, 0, 1, 1, 0],
    'c4': [1, 2, 2, 3, 3, 2],
    'Class': ['a', 'a', 'a', 'b', 'c', 'b']
})
model = KGMGFS(data=df)
model.run()
