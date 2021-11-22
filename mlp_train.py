import torch as th
from utils import masked_loss, masked_acc
import handle_data
from handle_data import *
import torch.optim
import pickle
import argparse
from utils import save_result

def accuracy(output, labels):
    length = len(output)
    kk = 0
    for i in range(len(output)):
        if output[i] == labels[i]:
            kk += 1
    return kk/length


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help='The random seed')
parser.add_argument('--threshold', type=float, default=0.4, help='The dtw threshold')
parser.add_argument('--dtw_initial_time', type=float, default=0, help='The dtw_initial_time')
parser.add_argument('--sim_time', default=3600,type=float, help="The dtw_computing_time")
parser.add_argument('--model', default='gcn')
parser.add_argument('--noise_level', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=5e-4) #（权重衰减）：目的就是为了让权重减少到更小的值，在一定程度上减少模型过拟合的问题
parser.add_argument('--early_stopping', type=int, default=15)
parser.add_argument('--max_degree', type=int, default=3) # K阶的切比雪夫近似矩阵的参数k
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
# parser.add_argument('--id_feature', type=bool, default=False)
args = parser.parse_args()


class MLP(th.nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim):
        super(MLP ,self).__init__()
        self.fc1 = th.nn.Linear(input_dim,hidden_dim)
        self.fc2 = th.nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = th.nn.Linear(hidden_dim,output_dim)

    def forward(self ,din):
        din = din.view(-1, 5)
        din = th.nn.functional.dropout(din,args.dropout)
        dout = th.nn.functional.relu(self.fc1(din))
        dout = th.sigmoid(self.fc2(dout))
        # dout = th.nn.functional.sigmoid(self.fc3(dout))
        return dout


model = MLP(5,32,2)

def generate_dtw_with_true_label(args, qb_information,unit_information):
    np.random.seed(args.seed)
    handle = Handle_data(qb_information,sim_time=args.sim_time,noise_level=args.noise_level)
    save_name = os.path.join("save_networkx", str(qb_information[-8:-4] + "_" + str(args.threshold) + "_" + str(args.dtw_initial_time) + "_" + str(args.sim_time) + ".pb"))
    if os.path.exists(save_name): ###避免每次加载，但是修改了就要注意名称。
        with open(save_name, "rb") as f:
            nx_g = pickle.load(f)
        current_type = handle.generate_current_type()
    else:
        nx_g = handle.generate_dtw_G(threshold=args.threshold,dtw_initial_time=args.dtw_initial_time)
        with open(save_name, "wb") as f:
            pickle.dump(nx_g,f)
        current_type = handle.current_type
    Red = handle_data.Handle_test_label(unit_information)
    nx_g = Red.obtain_test_label(nx_g, current_type=current_type)
    handle.draw(nx_g,is_dtw=True,is_show=True)
    return nx_g


# qb_information = r"C:\Users\zhouxianchen\PycharmProjects\deeplearning\logdata2\logdata\blue_qb_202011242026.txt"
# unit_information = r"C:\Users\zhouxianchen\PycharmProjects\deeplearning\logdata2\logdata\red_units_202011242026.txt"
# qb_information = r"C:\Users\zhouxianchen\PycharmProjects\deeplearning\logdata2\taskB\blue_qb_202012011041.txt"
# unit_information = r"C:\Users\zhouxianchen\PycharmProjects\deeplearning\logdata2\taskB\red_units_202012011041.txt"
qb_information = "./logdata2/logdata/blue_qb_202011242026.txt"
unit_information = "./logdata2/logdata/red_units_202011242026.txt"
G = generate_dtw_with_true_label(args, qb_information, unit_information)
model.train()
optimizer = torch.optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
# epoch = 200
print("nodes",len(G.nodes))
for i in range(args.epochs):
    loss = 0
    for v in G.nodes:
        if G.nodes[v]['label']!= 0:
            optimizer.zero_grad()
            input_tensor = th.tensor([G.nodes[v]['z'],  G.nodes[v]['y'] ,G.nodes[v]['z'],eval(G.nodes[v]['hx']),eval(G.nodes[v]['sp'])])
            if G.nodes[v]['real']==1:
                label_tensor = th.tensor([0])
            else:
                label_tensor= th.tensor([1])
            loss_fcn = th.nn.CrossEntropyLoss()
            output = model(input_tensor)
            loss += loss_fcn(output,label_tensor)
    loss.backward()
    optimizer.step()
    print(loss)

model.eval()
compute_real = []
real = []
for v in G.nodes:
    if G.nodes[v]['label']==0:
        input_tensor = th.tensor(
            [G.nodes[v]['z'], G.nodes[v]['y'], G.nodes[v]['z'], eval(G.nodes[v]['hx']), eval(G.nodes[v]['sp'])])
        output = model(input_tensor)
        label = th.argmax(output)
        compute_real.append(label)
        real.append(G.nodes[v]['real']-1)
print(compute_real)
print(real)
acc = accuracy(compute_real,real)
print(accuracy(compute_real,real))



save_result(acc,"save_acc/save_result_mlp.txt")
