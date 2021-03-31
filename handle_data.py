import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse
from dtw import dtw
from sklearn import preprocessing
from data import load_data, preprocess_features, preprocess_adj
import os
import torch
"""
这个文档是用来处理数据的， sim_time是选择相关的帧数，相关文件在logdata文件夹

选择 从开始时间到当前时间的轨迹，画出。
"""
class Handle_data:
    def __init__(self,filename, sim_time=3000,noise_level=0):#3973.2/3888.3
        self.qb_filename = filename
        self.qb_frames = self.read_file(self.qb_filename,noise_level=noise_level)
        self.frames = self.qb_frames
        self.current_time = self.obtain_nearest_frame(sim_time)
        print(self.current_time)
        self.frame = self.get_one_frame_data(current_time=self.current_time)


    def obtain_nearest_frame(self,sim_time):
        time_list = []
        for frame in self.frames:
            time_list.append(frame['sim_time'])
        return min(time_list, key=lambda x: abs(eval(x)-sim_time))




    @staticmethod
    def read_file(filename,noise_level=0):
        with open(filename, 'r') as f:
            kk = f.readlines()
        frames=[]
        for data in kk:
            node = {}
            single_data = data.split(',')
            node['sim_time'] = single_data[0]
            node['id'] = single_data[1]
            node['x'] = str(eval(single_data[2])+np.random.randn()*noise_level*2000)
            node['y'] = str(eval(single_data[3])+np.random.randn()*noise_level*2000)
            node['z'] = str(eval(single_data[4]))
            node['jb'] = single_data[5]
            node['hx'] = str(eval(single_data[6])+np.random.randn()*noise_level*10)
            node['sp'] = str(eval(single_data[7])+np.random.randn()*noise_level*10)
            node['lx'] = single_data[8]
            node['xh'] = single_data[9]
            node['wh'] = single_data[10]
            node['da'] = single_data[11][0:-1]
            if node['lx'] in ['11','18','15'] and node['z']!='10000':
                frames.append(node)
        return frames

    def get_single_position_sequence(self, id, dtw_initial_time):
        seq = []
        frames_sorted = sorted(self.frames,key=lambda x: x['sim_time'])
        for frame in frames_sorted:
            if frame['id'] == id and (eval(frame['sim_time']) <= eval(self.current_time)) and eval(frame['sim_time'])> dtw_initial_time:
                seq.append((eval(frame['x']),eval(frame['y']), eval(frame['y'])))
        return seq

    def get_all_squence(self, dtw_initial_time):
        node_id = []
        all_seq = {}
        for frame in self.frames:
            node_id.append(frame['id'])
        print("total number", len(set(node_id)))
        for id in list(set(node_id)):
            all_seq[str(id)] = self.get_single_position_sequence(id, dtw_initial_time)
        return all_seq

    def get_one_frame_data(self, current_time):
        using_frame = []
        for frame in self.frames:
            if frame['sim_time']== current_time:
                using_frame.append(frame)
  
        return using_frame


    def compute_dtw(self,all_seq, id_x, id_y):
        x = all_seq[id_x]
        y = all_seq[id_y]
        distance = lambda x, y: np.sqrt(np.abs(x[0]-y[0])**2+np.abs(x[1]-y[1])**2+np.abs(x[2]-y[2])**2)
        d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=distance)
        return d



    def generate_G(self, dist=80000):
        G = nx.Graph()
        self.current_type = [0]
        for node in self.frame:
            if node['xh'] != 'unknown':
                if node['xh'] not in self.current_type:
                    self.current_type.append(node['xh'])
                node['label'] = int(self.current_type.index(node['xh']))
            else:
                node['label']=0

            print("The label", node['label'])
            G.add_node(node['id'], x=eval(node['x']), y=eval(node['y']), z=eval(node['z']),
                       jb=node['jb'],hx=node['hx'],sp=node['sp'],lx=node['lx'],xh=node['xh'],wh=node['wh'],da=node['da'],label=node['label'], real = node['label'])
        for v in G.nodes:
            for u in G.nodes:
                if np.sqrt((G.nodes[v]['x']-G.nodes[u]['x'])**2+(G.nodes[v]['y']-G.nodes[u]['y'])**2)<dist:
                    G.add_edge(v,u)
        print(G.number_of_nodes())
        print(G.number_of_edges())
        return G

    def generate_current_type(self):
        self.current_type = [0]
        for node in self.frame:
            if node['xh'] != 'unknown':
                if node['xh'] not in self.current_type:
                    self.current_type.append(node['xh'])
        return self.current_type

    def generate_dtw_G(self, threshold=0.2, dtw_initial_time=0):
        """
        生成dtw权重的图G
        :return:
        """
        G = nx.Graph()
        self.current_type = [0]
        for node in self.frame:
            if node['xh'] != 'unknown':
                if node['xh'] not in self.current_type:
                    self.current_type.append(node['xh'])
                node['label'] = int(self.current_type.index(node['xh']))
            else:
                node['label']=0

            print("The label", node['label'])
            G.add_node(node['id'], x=eval(node['x']), y=eval(node['y']), z=eval(node['z']),
                       jb=node['jb'],hx=node['hx'],sp=node['sp'],lx=node['lx'],xh=node['xh'],wh=node['wh'],da=node['da'],label=node['label'], real=node['label'])
        all_seq = self.get_all_squence(dtw_initial_time=dtw_initial_time)
        #计算几率
        for v in G.nodes:
            for u in G.nodes:
                # if np.sqrt((G.nodes[v]['x']-G.nodes[u]['x'])**2+(G.nodes[v]['y']-G.nodes[u]['y'])**2)<100000:
                dist = self.compute_dtw(all_seq=all_seq, id_x=v, id_y=u)
                G.add_edge(v,u, weight=dist)
        adj_matrix = nx.to_numpy_matrix(G)
        adj_matrix = preprocessing.minmax_scale(adj_matrix)
        print(adj_matrix)
        i=0
        ###归一化边的权重
        edge_List = []
        for v in G.nodes:
            j=0
            for u in G.nodes:
                G[v][u]['weight']=adj_matrix[i][j]
                if adj_matrix[i][j]>threshold:
                    edge_List.append((v, u))
                j+=1
            i+=1
        ###移除大于0.5权重的边
        G.remove_edges_from(edge_List)
        print(G.number_of_nodes())
        print(G.number_of_edges())
        return G



    def draw(self,G, is_dtw=False, is_show=False):
        pos = {}
        color_label = []
        for v in G.nodes:
            # print(node)
            pos[v]=np.array([G.nodes[v]['x'], G.nodes[v]['y']])
            if G.nodes[v]['label'] == 1:
                color_label.append('b')
            elif G.nodes[v]['label']== 2:
                color_label.append('r')
            elif G.nodes[v]['label']== 0 and G.nodes[v]['real'] ==1 :
                color_label.append('g')
            elif G.nodes[v]['label']== 0 and G.nodes[v]['real'] ==2 :
                color_label.append('m')
            else:
                color_label.append('y')
        label_list = dict((n,str(i)) for i,n in enumerate(G.nodes))

        nx.draw(G,pos=pos,node_color=color_label, labels=label_list,with_labels=True)
        # edge_labels = nx.get_edge_attributes(G,'weight')
        # nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        if is_dtw:
            plt.savefig('save_fig/'+str(self.current_time)+'_dtw_.png')
        else:
            plt.savefig('save_fig/' + str(self.current_time) + 'dist'  + '.png')
        if is_show:
            plt.show()


def processing_data_generating(G):
    adj = nx.to_numpy_matrix(G)
    feature = np.zeros([G.number_of_nodes(), 2])
    labels = np.zeros(G.number_of_nodes(),dtype=int)
    idx_train = []
    idx_test = []
    i=0
    for v in G.nodes:
        feature[i] = np.array([G.nodes[v]['hx'], G.nodes[v]['sp']])
        if G.nodes[v]['label'] == 0:
            idx_test.append(i)
        else:
            idx_train.append(i)
        labels[i] = G.nodes[v]['real']
        i += 1
    features = preprocess_features(feature)
    i = torch.from_numpy(features[0].astype(np.int64)).long()
    v = torch.from_numpy(features[1])
    feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to_dense().to(torch.float32)
    return adj, feature,labels,np.array(idx_train),np.array(idx_test)


def data_for_mygat_training(G):
    adj, features, labels, idx_train, idx_test = processing_data_generating(G)
    k = len(labels)
    adj = scipy.sparse.csr_matrix(adj)
    train_mask = np.zeros(k, dtype=bool)
    train_mask[idx_train] = True
    test_mask = np.zeros(k, dtype=bool)
    test_mask[idx_test] = True
    labels = labels-1 ###要求label是[0,1]这样从0开始计数的
    return adj, features, labels, train_mask, test_mask


def data_for_gcn_training(G):
    """
    y_train和y_test 用来训练gcn的
    :param G:
    :return:
    """
    adj, features,labels,idx_train,idx_test = processing_data_generating(G)
    k = len(labels)
    l = max(labels)
    adj = scipy.sparse.csr_matrix(adj)
    train_mask = np.zeros(k, dtype=bool)
    train_mask[idx_train] = True
    test_mask = np.zeros(k, dtype=bool)
    test_mask[idx_test] = True
    labels = labels-1
    y_test = np.eye(k,l)[labels]
    y_test[idx_train, :] = 0
    y_train = np.eye(k,l)[labels]
    y_train[idx_test, :] = 0
    return adj, features, y_train, y_test, train_mask, test_mask

def data_for_rogcn_training(G):
    """
    y_train和y_test 用来训练gcn的
    这里的label有问题
    :param G:
    :return:
    """
    adj, features,labels,idx_train,idx_test = processing_data_generating(G)
    k = len(labels)
    l = max(labels)
    adj = scipy.sparse.csr_matrix(adj)
    train_mask = np.zeros(k, dtype=bool)
    train_mask[idx_train] = True
    test_mask = np.zeros(k, dtype=bool)
    test_mask[idx_test] = True
    labels = labels-1
    y_test = np.eye(k,l)[labels]
    y_test[idx_train, :] = 0
    y_train = np.eye(k,l)[labels]
    y_train[idx_test, :] = 0
    return adj, features,labels,idx_train,idx_test


class Red_handle(Handle_data):
    """
    用来获取test的真实标签
    """
    def __init__(self,filename):
        super().__init__(filename)

    def get_test_label(self,G):
        idx_test_real = []

        for v in G.nodes:
            for node in self.frame:
                if node['id'] == v:
                    idx_test_real.append(node['xh'])
        print("test real:", idx_test_real)
        print(len(idx_test_real))



class Handle_test_label(Handle_data):
    "寻找未知飞机真实的标签"

    def __init__(self,filename):
        self.filename = filename
        self.unit_frames = self.read_file(self.filename)


    def obtain_test_label(self, G,current_type):
        for v in G.nodes:
            if G.nodes[v]['label'] == 0:
                for node in self.unit_frames:
                    if node['id'] == v:
                        G.nodes[v]['real'] = current_type.index(node['xh'])

        return G


# 



if __name__ == "__main__":
    # qb_information = "C:/Users/zhouxianchen/Desktop/zxc/logdata/blue_qb_202009031945.txt"
    # units_infomation = "C:/Users/zhouxianchen/Desktop/zxc/logdata/blue_units_202009031945.txt"
    # redqb_information = "C:/Users/zhouxianchen/Desktop/zxc/logdata/red_qb_202009031945.txt"
    # unitsqb_information = "C:/Users/zhouxianchen/Desktop/zxc/logdata/red_qb_202009031945.txt"
    #
    # handle = Handle_data(qb_information, units_infomation)
    # seq = handle.get_all_squence()
    # G = handle.generate_dtw_G()
    # handle.draw(G)
    # Handle_test_label