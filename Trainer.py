# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com

from model import *
import sys
from torch import optim
import math
import pandas as pd
from termcolor import colored, cprint

from metrics import*
import wandb
from datetime import datetime
import tqdm
from scipy import signal


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list, offline_mode=False, tau=16, lambd=0.15, task="gestures", device="cuda",
                 network='MS-TCN2',debagging=False):

        if network == 'MS-TCN2':
            self.model = MST_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list,offline_mode=offline_mode)
        elif network == 'MS-TCN2_ISR':
            self.model = MST_TCN2_ISR(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list,offline_mode=offline_mode)
        elif network == 'LSTM':
            self.model = MT_RNN_dp("LSTM", input_dim=dim, hidden_dim=64, num_classes_list=num_classes_list,
                                bidirectional=offline_mode, dropout=0.4,num_layers=3)

        elif network == 'GRU':
            self.model = MT_RNN_dp("GRU", input_dim=dim, hidden_dim=128, num_classes_list=num_classes_list,
                                bidirectional=offline_mode, dropout=0.3,num_layers=3)
        elif network == 'MS-LSTM-TCN':
            self.model = RNN_MST_TCN2("LSTM",256, num_layers_R, num_R, num_f_maps, dim, num_classes_list, offline_mode,dropout_TCN=0.5,dropout_RNN=0.9)
        elif network == 'MS-TCN-LSTM':
            self.model = MST_TCN2_RNN(RNN_type='LSTM',num_layers_PG=num_layers_PG,hidden_dim=256,num_R= num_R,num_f_maps= num_f_maps, dim=dim, num_classes_list=num_classes_list, offline_mode=offline_mode)
        elif network == 'MS-GRU-TCN':
            self.model = RNN_MST_TCN2("GRU",256, num_layers_R, num_R, num_f_maps, dim, num_classes_list, offline_mode,dropout_TCN=0.5,dropout_RNN=0.9)
        elif network == 'MS-TCN-GRU':
            self.model = MST_TCN2_RNN(RNN_type='GRU',num_layers_PG=num_layers_PG,hidden_dim=256,num_R= num_R,num_f_maps= num_f_maps, dim=dim, num_classes_list=num_classes_list, offline_mode=offline_mode)

        else:
            raise NotImplemented
        self.debagging =debagging
        self.network = network
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss()
        self.num_classes_list = num_classes_list
        self.tau = tau
        self.lambd = lambd
        self.task =task


    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, eval_dict, args):

        number_of_seqs = len(batch_gen.list_of_train_examples)
        number_of_batches = math.ceil(number_of_seqs / batch_size)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + args.split)

        if args.upload is True:
            wandb.init(project=args.project, group= args.group,
                       name="split: " + args.split,
                       reinit=True)
            delattr(args, 'split')
            wandb.config.update(args)

        self.model.train()
        self.model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct1 = 0
            total1 = 0
            correct2 = 0
            total2 = 0
            correct3 = 0
            total3 = 0

            while batch_gen.has_next():
                if self.task == "multi-taks":
                    batch_input, batch_target_left, batch_target_right, batch_target_gestures, mask = batch_gen.next_batch(
                        batch_size)
                    batch_input, batch_target_left, batch_target_right, batch_target_gestures, mask = batch_input.to(
                        self.device), batch_target_left.to(self.device), batch_target_right.to(
                        self.device), batch_target_gestures.to(self.device), mask.to(self.device)

                elif self.task == "tools":
                    batch_input, batch_target_left, batch_target_right, mask = batch_gen.next_batch(batch_size)
                    batch_input, batch_target_left, batch_target_right, mask = batch_input.to(self.device), batch_target_left.to(
                        self.device), batch_target_right.to(self.device), mask.to(self.device)
                else:
                    batch_input, batch_target_gestures, mask = batch_gen.next_batch(batch_size)
                    batch_input, batch_target_gestures, mask = batch_input.to(self.device), batch_target_gestures.to(
                        self.device), mask.to(self.device)

                optimizer.zero_grad()
                predictions1, predictions2, predictions3 =[],[],[]
                if self.task == "multi-taks":
                    if self.network == "LSTM" or self.network == "GRU":
                        lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                        predictions1, predictions2, predictions3 = self.model(batch_input, lengths)
                        predictions1 = (predictions1 * mask).unsqueeze_(0)
                        predictions2 = (predictions2 * mask).unsqueeze_(0)
                        predictions3 = (predictions3 * mask).unsqueeze_(0)

                    else:
                        predictions1, predictions2, predictions3 = self.model(batch_input)

                elif self.task == "tools":
                    if self.network == "LSTM" or self.network == "GRU":
                        lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                        predictions2, predictions3 = self.model(batch_input, lengths)
                        predictions2 = (predictions2 * mask).unsqueeze_(0)
                        predictions3 = (predictions3 * mask).unsqueeze_(0)

                    else:
                        predictions2, predictions3 = self.model(batch_input)
                else:
                    if self.network == "LSTM" or self.network == "GRU":
                        lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                        predictions1 = self.model(batch_input, lengths)
                        predictions1 = (predictions1[0] * mask).unsqueeze_(0)
                    else:
                        predictions1 = self.model(batch_input)[0]

                loss = 0
                for p in predictions1:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[0]),
                                    batch_target_gestures.view(-1))
                    if self.network not in ["GRU","LSTM"]:
                        loss += self.lambd * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=self.tau))

                for p in predictions2:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[1]),
                                    batch_target_right.view(-1))
                    if self.network not in ["GRU","LSTM"]:
                        loss += self.lambd * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=self.tau))

                for p in predictions3:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_list[1]),
                                    batch_target_left.view(-1))
                    if self.network not in ["GRU","LSTM"]:
                        loss += self.lambd * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=self.tau))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                if self.task == "multi-taks" or self.task == "gestures":
                    _, predicted1 = torch.max(predictions1[-1].data, 1)
                    correct1 += ((predicted1 == batch_target_gestures).float().squeeze(1)).sum().item()
                    total1 += predicted1.shape[1]

                if self.task == "multi-taks" or self.task == "tools":

                    _, predicted2 = torch.max(predictions2[-1].data, 1)
                    _, predicted3 = torch.max(predictions3[-1].data, 1)
                    correct2 += ((predicted2 == batch_target_right).float().squeeze(1)).sum().item()
                    total2 += predicted2.shape[1]
                    correct3 += ((predicted3 == batch_target_left).float().squeeze(1)).sum().item()
                    total3 += predicted3.shape[1]

                pbar.update(1)

            batch_gen.reset()
            pbar.close()
            if not self.debagging:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if self.task == "multi-taks":
                print(colored(dt_string, 'green', attrs=[
                    'bold']) + "  " + "[epoch %d]: train loss = %f,  train acc gesture = %f,  train acc right= %f,  train acc left = %f" % (
                      epoch + 1,
                      epoch_loss / len(batch_gen.list_of_train_examples),
                      float(correct1) / total1, float(correct2) / total2, float(correct3) / total3))

                train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                                 "train acc left": float(correct1) / total1, "train acc right": float(correct2) / total2,
                                 "train acc gestures": float(correct3) / total3}
            elif self.task == "tools":
                print(colored(dt_string, 'green', attrs=[
                    'bold']) + "  " + "[epoch %d]: train loss = %f,   train acc right = %f,  train acc left = %f" % (
                          epoch + 1,
                          epoch_loss / len(batch_gen.list_of_train_examples),
                          float(correct2) / total2, float(correct3) / total3))
                train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                                 "train acc left": float(correct2) / total2,
                                 "train acc right": float(correct3) / total3}
            else:
                print(colored(dt_string, 'green',
                              attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                          epoch_loss / len(
                                                                                                              batch_gen.list_of_train_examples),
                                                                                                          float(
                                                                                                              correct1) / total1))
                train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                                 "train acc": float(correct1) / total1}

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['reverse', 'bold']))
                results = {"epoch": epoch}
                results.update(self.evaluate(eval_dict, batch_gen))
                eval_results_list.append(results)
                if args.upload is True:
                    wandb.log(results)

        return eval_results_list, train_results_list

    def evaluate(self, eval_dict, batch_gen):
        results = {}
        device = eval_dict["device"]
        features_path = eval_dict["features_path"]
        sample_rate = eval_dict["sample_rate"]
        actions_dict = eval_dict["actions_dict_tools"]
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_right = eval_dict["gt_path_tools_right"]
        ground_truth_path_left = eval_dict["gt_path_tools_left"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            list_of_vids = batch_gen.list_of_valid_examples
            recognition1_list = []
            recognition2_list = []
            recognition3_list = []

            for seq in list_of_vids:
                # print vid
                features = np.load(features_path + seq.split('.')[0] + '.npy')
                if batch_gen.normalization == "Min-max":
                    numerator = features.T - batch_gen.min
                    denominator = batch_gen.max - batch_gen.min
                    features = (numerator / denominator).T
                elif batch_gen.normalization == "Standard":
                    numerator = features.T - batch_gen.mean
                    denominator = batch_gen.std
                    features = (numerator / denominator).T
                elif batch_gen.normalization == "samplewise_SD":
                    samplewise_meam = features.mean(axis=1)
                    samplewise_std = features.std(axis=1)
                    numerator = features.T - samplewise_meam
                    denominator = samplewise_std
                    features = (numerator / denominator).T

                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                if self.task == "multi-taks":
                    predictions1, predictions2, predictions3 = self.model(input_x)
                elif self.task == "tools":
                    if self.network == "LSTM" or self.network == "GRU":
                        predictions2, predictions3 = self.model(input_x, torch.tensor([features.shape[1]]))
                        predictions2 = predictions2.unsqueeze_(0)
                        predictions2 = torch.nn.Softmax(dim=2)(predictions2)
                        predictions3 = predictions3.unsqueeze_(0)
                        predictions3 = torch.nn.Softmax(dim=2)(predictions3)



                    else:
                        predictions2, predictions3 = self.model(input_x)
                else:
                    if self.network == "LSTM" or self.network == "GRU":
                        predictions1 = self.model(input_x, torch.tensor([features.shape[1]]))
                        predictions1 = predictions1[0].unsqueeze_(0)
                        predictions1 = torch.nn.Softmax(dim=2)(predictions1)
                    else:
                        predictions1 = self.model(input_x)[0]

                if self.task == "multi-taks" or self.task == "gestures":
                    _, predicted1 = torch.max(predictions1[-1].data, 1)
                    predicted1 = predicted1.squeeze()

                if self.task == "multi-taks" or self.task == "tools":
                    _, predicted2 = torch.max(predictions2[-1].data, 1)
                    _, predicted3 = torch.max(predictions3[-1].data, 1)
                    predicted2 = predicted2.squeeze()
                    predicted3 = predicted3.squeeze()

                recognition1 = []
                recognition2 = []
                recognition3 = []
                if self.task == "multi-taks" or self.task == "gestures":
                    for i in range(len(predicted1)):
                        recognition1 = np.concatenate((recognition1, [list(actions_dict_gesures.keys())[
                                                                          list(actions_dict_gesures.values()).index(
                                                                              predicted1[i].item())]] * sample_rate))
                    recognition1_list.append(recognition1)
                if self.task == "multi-taks" or self.task == "tools":

                    for i in range(len(predicted2)):
                        recognition2 = np.concatenate((recognition2, [list(actions_dict.keys())[
                                                                          list(actions_dict.values()).index(
                                                                              predicted2[i].item())]] * sample_rate))
                    recognition2_list.append(recognition2)

                    for i in range(len(predicted3)):
                        recognition3 = np.concatenate((recognition3, [list(actions_dict.keys())[
                                                                          list(actions_dict.values()).index(
                                                                              predicted3[i].item())]] * sample_rate))
                    recognition3_list.append(recognition3)
            if self.task == "multi-taks" or self.task == "gestures":

                print("gestures results")
                results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                                 recognition_list=recognition1_list, list_of_videos=list_of_vids,
                                                 suffix="gesture")
                results.update(results1)

            if self.task == "multi-taks" or self.task == "tools":

                print("right hand results")
                results2, _ = metric_calculation(ground_truth_path=ground_truth_path_right,
                                                 recognition_list=recognition2_list, list_of_videos=list_of_vids,
                                                 suffix="right")
                print("left hand results")
                results3, _ = metric_calculation(ground_truth_path=ground_truth_path_left,
                                                 recognition_list=recognition3_list, list_of_videos=list_of_vids,
                                                 suffix="left")
                results.update(results2)
                results.update(results3)

            self.model.train()
            return results




