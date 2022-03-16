# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
import torch
from Trainer import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import pandas as pd
from datetime import datetime
from termcolor import colored, cprint
import random
import time



dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['APAS'], default="APAS")
parser.add_argument('--task',choices=['gestures','tools','multi-taks'], default="tools")
parser.add_argument('--network',choices=['MS-TCN2','LSTM','GRU'], default="MS-TCN2")
parser.add_argument('--split',choices=['0', '1', '2', '3','4', 'all'], default='all')
parser.add_argument('--features_dim', default='36', type=int)
parser.add_argument('--lr', default='0.00110532', type=float)
parser.add_argument('--num_epochs', default=40, type=int)

# Architectuyre
parser.add_argument('--eval_rate', default=1, type=int)
parser.add_argument('--num_f_maps', default='128', type=int)

parser.add_argument('--num_layers_PG', default=13, type=int)
parser.add_argument('--num_layers_R',default=13, type=int)
parser.add_argument('--normalization', choices=['Min-max', 'Standard', 'samplewise_SD','none'], default='Standard', type=str)
parser.add_argument('--num_R', default=1, type=int)

parser.add_argument('--hidden_dim_rnn', default=256, type=int)
parser.add_argument('--num_layers_rnn', default=3, type=int)
parser.add_argument('--sample_rate', default=1, type=int)


parser.add_argument('--loss_tau',default=16, type=float)
parser.add_argument('--loss_lambda', default=0.60755946, type=float)
parser.add_argument('--dropout', default=0.590357869, type=float)

parser.add_argument('--offline_mode', default=True, type=bool)
parser.add_argument('--project', default="Offline RNN nets Sensor paper Final", type=str)
parser.add_argument('--group', default=dt_string + " ", type=str)
parser.add_argument('--use_gpu_num',default ="0", type=str )
parser.add_argument('--upload', default=False, type=bool)
parser.add_argument('--filtered_data', default=True, type=bool)
parser.add_argument('--debagging', default=False, type=bool)
parser.add_argument('--hyper_parameter_tuning', default=False, type=bool)


args = parser.parse_args()
debagging = args.debagging
if debagging:
    args.upload = False


print(args)
seed = int(time.time())
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# use the full temporal resolution @ 30Hz
if args.network in ["GRU","LSTM"]:
    sample_rate = args.sample_rate
    bz = 5

else:
    sample_rate = args.sample_rate
    bz = 1




list_of_splits =[]
if len(args.split) == 1:
    list_of_splits.append(int(args.split))

elif args.dataset == "APAS":
    list_of_splits = list(range(0,5))
else:
    raise NotImplemented

loss_lambda = args.loss_lambda
loss_tau = args.loss_tau
num_epochs = args.num_epochs
eval_rate = args.eval_rate
features_dim = args.features_dim
lr = args.lr
offline_mode = args.offline_mode
num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps
experiment_name = args.group +" task:"  + args.task + " splits: " + args.split +" net: " + args.network + " is Offline: " + str(args.offline_mode)
args.group = experiment_name
hyper_parameter_tuning = args.hyper_parameter_tuning
print(colored(experiment_name, "green"))


summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
if not debagging:
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)


full_eval_results = pd.DataFrame()
full_train_results = pd.DataFrame()
full_test_results = pd.DataFrame()


for split_num in list_of_splits:
    print("split number: " + str(split_num))
    args.split = str(split_num)

    folds_folder = os.path.join("data",args.dataset,"folds")

    if args.dataset == "APAS":
        if args.filtered_data:
            features_path =  os.path.join("data",args.dataset,"kinematics_with_filtration_npy")

        else:
            features_path =  os.path.join("data", args.dataset,"kinematics_without_filtration_npy")

    else:
        features_path = os.path.join("data", args.dataset,"kinematics_npy")

    gt_path_gestures = os.path.join("data", args.dataset,"transcriptions_gestures")
    gt_path_tools_left= os.path.join("data", args.dataset, "transcriptions_tools_left")
    gt_path_tools_right = os.path.join("data", args.dataset, "transcriptions_tools_right")

    mapping_gestures_file = os.path.join("data", args.dataset, "mapping_gestures.txt")

    mapping_tool_file = os.path.join("data", args.dataset, "mapping_tools.txt")


    model_dir = os.path.join("models", args.dataset,experiment_name,"split" +args.split)


    if not debagging:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    file_ptr = open(mapping_gestures_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict_gestures = dict()
    for a in actions:
        actions_dict_gestures[a.split()[1]] = int(a.split()[0])
    num_classes_tools =0
    actions_dict_tools = dict()
    if args.dataset == "APAS":
        file_ptr = open(mapping_tool_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            actions_dict_tools[a.split()[1]] = int(a.split()[0])
        num_classes_tools = len(actions_dict_tools)

    num_classes_gestures = len(actions_dict_gestures)

    if args.task == "gestures":
        num_classes_list =[num_classes_gestures]
    elif args.dataset == "APAS" and args.task == "tools":
        num_classes_list = [num_classes_tools,num_classes_tools]
    elif args.dataset == "APAS" and args.task == "multi-taks":
        num_classes_list=[num_classes_gestures, num_classes_tools, num_classes_tools]

    trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes_list, offline_mode=offline_mode, tau=loss_tau, lambd=loss_lambda,hidden_dim_rnn=args.hidden_dim_rnn,num_layers_rnn=args.num_layers_rnn,dropout=args.dropout,task=args.task,device=device,network=args.network,hyper_parameter_tuning=hyper_parameter_tuning,debagging=debagging)
    print(gt_path_gestures)
    batch_gen = BatchGenerator(num_classes_gestures,num_classes_tools, actions_dict_gestures,actions_dict_tools,features_path,split_num,folds_folder ,gt_path_gestures, gt_path_tools_left, gt_path_tools_right, sample_rate=sample_rate,normalization=args.normalization,task=args.task)
    eval_dict ={"features_path":features_path,"actions_dict_gestures": actions_dict_gestures, "actions_dict_tools":actions_dict_tools, "device":device, "sample_rate":sample_rate,"eval_rate":eval_rate,
                "gt_path_gestures":gt_path_gestures, "gt_path_tools_left":gt_path_tools_left, "gt_path_tools_right":gt_path_tools_right,"task":args.task}
    best_valid_results, eval_results, train_results, test_results = trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr,eval_dict=eval_dict,args=args)


    if not debagging:
        eval_results = pd.DataFrame(eval_results)
        train_results = pd.DataFrame(train_results)
        test_results = pd.DataFrame(test_results)
        eval_results["split_num"] = str(split_num)
        train_results["split_num"] = str(split_num)
        test_results["split_num"] = str(split_num)
        eval_results["seed"] = str(seed)
        train_results["seed"] = str(seed)
        test_results["seed"] = str(seed)

        full_eval_results = pd.concat([full_eval_results, eval_results], axis=0)
        full_train_results = pd.concat([full_train_results, train_results], axis=0)
        full_test_results = pd.concat([full_test_results, test_results], axis=0)
        full_eval_results.to_csv(summaries_dir+"/"+args.network +"_evaluation_results.csv",index=False)
        full_train_results.to_csv(summaries_dir+"/"+args.network +"_train_results.csv",index=False)
        full_test_results.to_csv(summaries_dir+"/"+args.network +"_test_results.csv",index=False)