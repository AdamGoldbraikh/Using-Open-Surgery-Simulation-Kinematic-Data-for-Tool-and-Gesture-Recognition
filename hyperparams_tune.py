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
import optuna
from optuna.samplers import RandomSampler

def calculate_mean_of_best(all_best):
    average={}
    metrics = list(all_best[0].keys())
    for metric in metrics:
        total =0
        for epoch in all_best:
            total+=epoch[metric]
        total = total / len(all_best)
        average[metric] = total
    return average
dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['APAS'], default="APAS")
parser.add_argument('--task',choices=['gestures','tools','multi-taks'], default="gestures")
parser.add_argument('--network',choices=['MS-TCN2','LSTM','GRU'], default="MS-TCN2")
parser.add_argument('--split',choices=['0', '1', '2', '3','4', 'all'], default='all')
parser.add_argument('--features_dim', default='36', type=int)
parser.add_argument('--lr', default='0.0005', type=float)
parser.add_argument('--num_epochs', default=40, type=int)

# Architectuyre
parser.add_argument('--eval_rate', default=1, type=int)
parser.add_argument('--num_f_maps', default='64', type=int)

parser.add_argument('--num_layers_PG', default=13, type=int)
parser.add_argument('--num_layers_R',default=13, type=int)
parser.add_argument('--normalization', choices=['Min-max', 'Standard', 'samplewise_SD','none'], default='Standard', type=str)
parser.add_argument('--num_R', default=3, type=int)
parser.add_argument('--hidden_dim_rnn', default=64, type=int)
parser.add_argument('--num_layers_rnn', default=3, type=int)

parser.add_argument('--loss_tau',default=16, type=float)
parser.add_argument('--loss_lambda', default=0.5, type=float)
parser.add_argument('--dropout', default=0.5, type=float)

parser.add_argument('--offline_mode', default=True, type=bool)
parser.add_argument('--project', default="hyper-parameters tuning GRU", type=str)
parser.add_argument('--group', default=dt_string + " ", type=str)
parser.add_argument('--use_gpu_num',default ="0", type=str )
parser.add_argument('--upload', default=False, type=bool)
parser.add_argument('--filtered_data', default=True, type=bool)
parser.add_argument('--debagging', default=False, type=bool)
parser.add_argument('--hyper_parameter_tuning', default=True, type=bool)


args = parser.parse_args()
debagging = args.debagging
if debagging:
    args.upload = False

seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# use the full temporal resolution @ 30Hz
if args.network in ["GRU","LSTM"]:
    bz = 5

else:
    sample_rate = 1
    bz = 1




list_of_splits =[]
if len(args.split) == 1:
    list_of_splits.append(int(args.split))

elif args.dataset == "APAS":
    list_of_splits = list(range(0,5))
else:
    raise NotImplemented
num_epochs = args.num_epochs
eval_rate = args.eval_rate
offline_mode = args.offline_mode
features_dim = args.features_dim


def objective(trial):



    if args.network == "MS-TCN2":
        sample_rate =1
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

        args.lr = lr

        dropout = trial.suggest_float("dropout", 0.45, 1, log=False)
        args.dropout = dropout

        num_layers = trial.suggest_int("num_layers",7,13,step=2)
        num_layers_PG = num_layers
        args.num_layers_PG = num_layers_PG

        num_layers_R = num_layers
        args.num_layers_R = num_layers_R
        num_R = trial.suggest_int("num_R",1,4)
        args.num_R = num_R
        num_f_maps = trial.suggest_categorical("num_f_maps",[16,24,32,64,128])
        args.num_f_maps =num_f_maps
        loss_tau = trial.suggest_categorical("loss_tau_power",[8,12,16,20])
        args.loss_tau = loss_tau
        loss_lambda = trial.suggest_float("loss_lambda", 0, 1, log=False)
        args.loss_lambda = loss_lambda
        hyper_param_dict = {"network": args.network,"trial":trial.number,"lr":lr,"dropout":dropout,"num_layers":num_layers,"num_f_maps":num_f_maps,"num_R":num_R,"loss_tau":loss_tau,"loss_lambda":loss_lambda}

    else:
        lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)

        args.lr = lr

        dropout = trial.suggest_float("dropout", 0, 1, log=False)
        args.dropout = dropout
        hidden_size_power = trial.suggest_int("hidden_size_power", 5, 9)
        sample_rate = trial.suggest_int("sample_rate",4, 8)

        hidden_dim_rnn = 2 ** hidden_size_power
        args.hidden_dim_rnn = hidden_dim_rnn
        num_layers_rnn = trial.suggest_int("num_layers_rnn", 1, 3)
        args.num_layers_rnn = num_layers_rnn

        hyper_param_dict = {"network": args.network, "trial":trial.number, "lr": lr, "dropout": dropout, "hidden_dim_rnn": hidden_dim_rnn,
                            "num_layers_rnn": num_layers_rnn,"sample_rate":sample_rate}

    print(colored(hyper_param_dict, "blue"))


    experiment_name = dt_string + " splits: " + args.split +" net: " + args.network  +  " trial number: " +str(trial.number)
    args.group = experiment_name
    hyper_parameter_tuning = args.hyper_parameter_tuning
    print(colored(experiment_name, "green"))
    print(colored(args, "red"))



    summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
    hyper_param_tuning = "./hyper_param_tuning/" + args.dataset + "/" + args.network +"_Random"
    if not debagging:
        if not os.path.exists(summaries_dir):
            os.makedirs(summaries_dir)
        if not os.path.exists(hyper_param_tuning):
            os.makedirs(hyper_param_tuning)



    full_eval_results = pd.DataFrame()
    full_train_results = pd.DataFrame()
    full_test_results = pd.DataFrame()

    all_best= []
    for split_num in list_of_splits:
        print("split number: " + str(split_num))
        args.split = str(split_num)

        folds_folder = os.path.join("data", args.dataset, "folds")

        if args.dataset == "APAS":
            if args.filtered_data:
                features_path = os.path.join("data", args.dataset, "kinematics_with_filtration_npy")

            else:
                features_path = os.path.join("data", args.dataset, "kinematics_without_filtration_npy")

        else:
            features_path = os.path.join("data", args.dataset, "kinematics_npy")

        gt_path_gestures = os.path.join("data", args.dataset, "transcriptions_gestures")
        gt_path_tools_left = os.path.join("data", args.dataset, "transcriptions_tools_left")
        gt_path_tools_right = os.path.join("data", args.dataset, "transcriptions_tools_right")
        mapping_gestures_file = os.path.join("data", args.dataset, "mapping_gestures.txt")
        mapping_tool_file = os.path.join("data", args.dataset, "mapping_tools.txt")

        model_dir = "./models/"+args.dataset+"/"+ experiment_name+"/split_"+args.split

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

        trainer = Trainer(args.num_layers_PG, args.num_layers_R, args.num_R, args.num_f_maps, args.features_dim, num_classes_list, offline_mode=args.offline_mode, tau=args.loss_tau, lambd=args.loss_lambda,hidden_dim_rnn=args.hidden_dim_rnn,num_layers_rnn=args.num_layers_rnn,dropout=args.dropout,task=args.task,device=device,network=args.network,hyper_parameter_tuning=args.hyper_parameter_tuning,debagging=args.debagging)

        batch_gen = BatchGenerator(num_classes_gestures,num_classes_tools, actions_dict_gestures,actions_dict_tools,features_path,split_num,folds_folder ,gt_path_gestures, gt_path_tools_left, gt_path_tools_right, sample_rate=sample_rate,normalization=args.normalization,task=args.task)
        eval_dict ={"features_path":features_path,"actions_dict_gestures": actions_dict_gestures, "actions_dict_tools":actions_dict_tools, "device":device, "sample_rate":sample_rate,"eval_rate":eval_rate,
                    "gt_path_gestures":gt_path_gestures, "gt_path_tools_left":gt_path_tools_left, "gt_path_tools_right":gt_path_tools_right,"task":args.task}
        best_valid_results, eval_results, train_results, test_results = trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr,eval_dict=eval_dict,args=args)
        all_best.append(best_valid_results)

        if not debagging:
            eval_results = pd.DataFrame(eval_results)
            train_results = pd.DataFrame(train_results)
            test_results = pd.DataFrame(test_results)
            eval_results = eval_results.add_prefix('split_'+str(split_num)+'_')
            train_results = train_results.add_prefix('split_'+str(split_num)+'_')
            test_results = test_results.add_prefix('split_'+str(split_num)+'_')
            full_eval_results = pd.concat([full_eval_results, eval_results], axis=0)
            full_train_results = pd.concat([full_train_results, train_results], axis=0)
            full_test_results = pd.concat([full_test_results, test_results], axis=0)
            full_eval_results.to_csv(summaries_dir+"/evaluation_results.csv",index=False)
            full_train_results.to_csv(summaries_dir+"/train_results.csv",index=False)
            full_test_results.to_csv(summaries_dir+"/test_results.csv",index=False)

    average_metrics = calculate_mean_of_best(all_best)
    hyper_param_dict.update(average_metrics)
    if not debagging:
        df = pd.DataFrame(list(hyper_param_dict.values())).transpose()
        df.columns = hyper_param_dict.keys()
        df.to_csv(hyper_param_tuning+"/trial_"+str(trial.number) +".csv",index=False)


    return average_metrics[list(average_metrics.keys())[3]]

# study = optuna.create_study(sampler=RandomSampler(), study_name="RandomTune_"+args.network,storage='sqlite:///myDB_Random_'+args.network+'db',load_if_exists=True ,direction="maximize")
study = optuna.create_study(sampler=RandomSampler(), study_name="RandomTuneA_"+args.network,storage='sqlite:///myDB_Random_'+args.network+'_A.db',load_if_exists=True ,direction="maximize")

print(f"Sampler is {study.sampler.__class__.__name__}")

study.optimize(objective, n_trials=55)

# optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])

print("summary")
print(colored(study.best_params, "green"))
print("trials")
print(colored(study.trials, "green"))


