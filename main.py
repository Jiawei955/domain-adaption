import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from datetime import datetime as dt

from model.CNN import Network
from dataloader import TrainSet, TestSet, InflSet, TrainSet_CSSA
import numpy as np

from pytorch_influence_functions.utils import get_default_config, init_logging, save_json
from pytorch_influence_functions.calc_influence_function import calc_img_wise
import json
import os
import statistics
from fine_tune import train_fine_tune


def save_model(net, path):
    torch.save(net.state_dict(), path)


def load_model(path):
    net = Network()
    net.load_state_dict(torch.load(path))
    net.cuda()
    return net


# Constrastive Semantic Alignment Loss
def csa_loss(x, y, class_eq):
    margin = 1
    dist = F.pairwise_distance(x, y)
    loss = class_eq * dist.pow(2)
    loss += (1 - class_eq) * (margin - dist).clamp(min=0).pow(2)
    return loss.mean()

def train_CSSA(device, net, loader, method):
    ce_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters())
    net.train()
    for i, (src_img, src_label, target_img, target_label) in enumerate(loader):
        src_img, target_img = (x.to(device, dtype=torch.float) for x in [src_img, target_img])
        src_label, target_label = (x.to(device, dtype=torch.long) for x in [src_label, target_label])
        src_pred, src_feature = net(src_img)
        target_pred, target_feature = net(target_img)

        ce  = ce_loss(src_pred, src_label)
        ce_t  = ce_loss(target_pred, target_label)
        csa = csa_loss(src_feature, target_feature, (src_label == target_label).float())

        if method == 'CSSA':
            loss = (1 - alpha) * ce + alpha * csa
        elif method == 'source':
            loss = ce
        elif method == 'source_target':
            loss = ce + ce_t
        else:
            raise
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss.item()

def train(device, net, loader):
    ce_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters())
    net.train()
    for i, (img, label) in enumerate(loader):
        img = img.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        pred, feature = net(img)
        loss = ce_loss(pred, label)
        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss.item()

def test(device, net, loader):
    correct = 0
    net.eval()
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            pred, _ = net(img)
            _, idx = pred.max(dim=1)
            correct += (idx == label).sum().cpu().item()
    acc = correct / len(loader.dataset)
    return acc

def run_trial(results, mode, domain_adaptation_task, sample_per_class, src_repetition, tgt_repetition, batch, epochs, retrain_epochs, alpha):

    if domain_adaptation_task in ['USPS_to_MNIST', 'MNIST_to_USPS']:
        class_num = 10

    config = get_default_config()
    if not os.path.exists(config['outdir']):
        os.mkdir(config['outdir'])

    device = torch.device("cuda")
    repetition = (src_repetition, tgt_repetition)

    results[str(repetition)] = {}
    train_set_base = TrainSet(domain_adaptation_task, 'baseline', src_repetition, tgt_repetition, sample_per_class)
    train_set_base_loader = DataLoader(train_set_base, batch_size=batch, shuffle=True)
    test_set = TestSet(domain_adaptation_task, 0, sample_per_class)
    test_set_loader = DataLoader(test_set, batch_size=batch, shuffle=True)
    print("Dataset Length Train (baseline) : ", len(train_set_base), " Test : ", len(test_set))

    if not os.path.exists('saved_models'):
        os.mkdir('saved_models')

    mplain_PATH = f'./saved_models/{domain_adaptation_task}_plain_s{str(sample_per_class)}'\
                  f'_r{str(repetition)}_b{str(batch)}_e{str(epochs)}_a{str(alpha)}'

    mbase_PATH = f'./saved_models/{domain_adaptation_task}_baseline_s{str(sample_per_class)}'\
                 f'_r{str(repetition)}_b{str(batch)}_e{str(epochs)}_a{str(alpha)}.pth'

    NEW_PATH = f'./saved_models/re_{domain_adaptation_task}_{mode}_s{str(sample_per_class)}'\
               f'_r{str(repetition)}_b{str(batch)}_e{str(epochs)}_a{str(alpha)}.pth'

    influence_path = config['outdir']+f'/influence_results_r{str(repetition)}.json'

    if mode in ['all','train_baseline']:
        results[str(repetition)]['baseline_acc'] = {}
        display = []
        sum_acc = 0
        for i in range(10):
            net = Network().to(device)

            best_test_acc = 0
            best_epoch = 0
            for epoch in range(epochs):
                train_loss = train(device, net, train_set_base_loader)
                test_acc = test(device, net, test_set_loader)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch

            results[str(repetition)]['baseline_acc'][i]=(best_epoch,best_test_acc)
            display.append(best_test_acc)
            print(f'repetition [{str(repetition)}] at time {str(i)}th training best acc at epoch'\
                    f'{str(best_epoch)}: {str(best_test_acc)}')
            sum_acc += best_test_acc

        results[str(repetition)]['baseline_acc']['avg_acc'] = sum_acc/10
        display.append(sum_acc/10)
        print(domain_adaptation_task, f' repetition [{str(repetition)}] avg test acc: ', sum_acc/10)
        for i in display:
            print(i)

    if mode in ['all','influence']:
        infl_src = InflSet(domain_adaptation_task, 'source', src_repetition, sample_per_class)
        infl_src_loader = DataLoader(infl_src, batch_size=batch)
        infl_tgt = InflSet(domain_adaptation_task, 'target', tgt_repetition, sample_per_class)
        infl_tgt_loader = DataLoader(infl_tgt, batch_size=batch)
        print("infl_src : ", len(infl_src), "infl_tgt : ", len(infl_tgt))

        results[str(repetition)]['plain_acc'] = {}
        results[str(repetition)]['influence'] = {}
        infl_sum = np.zeros([len(infl_tgt), len(infl_src)]) 
        for i in range(5):
            net = Network().to(device)
            best_acc = 0
            best_epoch = 0
            for epoch in range(epochs):
                train_loss = train(device, net, infl_src_loader)
                test_acc = test(device, net, test_set_loader)
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    save_model(net, mplain_PATH+f'v{str(i)}.pth')

            results[str(repetition)]['plain_acc'][i] = (best_epoch, best_acc)
            print(f'repetition [{str(repetition)}] at time {str(i)}th plain model best acc at epoch'\
                    f'{str(best_epoch)}: {str(best_acc)}')

            net = load_model(mplain_PATH+f'v{str(i)}.pth')

            # fine-tune the net with 10 target examples, spc = 1

            train_set = TestSet(domain_adaptation_task,1,1)
            val_set = TestSet(domain_adaptation_task,2,1)
            train_set_indices = np.random.permutation(len(train_set))[:len(train_set)]
            val_set_indices = np.random.permutation(len(val_set))[:100]
            train_loader = DataLoader(train_set,batch_size=1,shuffle=False,sampler=SubsetRandomSampler(train_set_indices))
            val_loader = DataLoader(val_set,batch_size=2,shuffle=False,sampler=SubsetRandomSampler(val_set_indices))
            print("fine_tuning the net...")
            train_fine_tune(net,train_loader,val_loader,200)

            #################

            infl_arr = calc_img_wise(config, net, infl_src_loader, infl_tgt_loader, i)
            # results[str(repetition)]['influence'][i] = infl_arr.tolist()
            infl_sum = np.add(infl_sum, infl_arr)

        infl_avg = infl_sum/5
        results[str(repetition)]['influence']['avg'] = infl_avg.tolist()
        for target in range(5):
            print("Results stats for target", target)
            acs = np.sort(infl_avg[target])
            print(acs[:10])
            print(acs[-10:])
            print("median", np.median(infl_avg[target]))
            print("mean", np.mean(infl_avg[target]))
            print("std", np.std(infl_avg[target]))

        save_json(infl_avg.tolist(), influence_path)
    
    if mode in ['all', 'stats']:
        
        with open(influence_path) as json_file:
            data = json.load(json_file)

        counter = np.zeros((8,len(data[0])))

        for i in range(sample_per_class*10):
            infl = data[i]
            std = statistics.stdev(infl)
            avg = np.mean(infl)
            print("std for target image",i,":",std)

            counter[0] += infl
            counter[1] += [1 if x > avg+2*std else -1 if x < avg-2*std else 0 for x in infl]
            counter[2] += [-1 if x < -1*std else 0 for x in infl]
            counter[3] += [-1 if x < -2*std else 0 for x in infl]
            counter[4] += [1 if x > std else 0 for x in infl]
            counter[5] += [-1 if abs(x) > 2*std else 0 for x in infl]


        # counter 6
        counter[6][counter[0]<np.percentile(counter[0],10)] = 1
        counter[6][(counter[0] >= np.percentile(counter[0], 10)) & (counter[0] < np.percentile(counter[0], 50)) ] = 2
        counter[6][(counter[0] >= np.percentile(counter[0], 50)) & (counter[0] <= np.percentile(counter[0], 90)) ] = 3
        counter[6][counter[0] > np.percentile(counter[0], 90)] = 5

        # counter 7
        counter[7][counter[0] < np.percentile(counter[0], 10)] = 1
        counter[7][(counter[0] >= np.percentile(counter[0], 10)) & (counter[0] < np.percentile(counter[0], 30))] = 2
        counter[7][(counter[0] >= np.percentile(counter[0], 30)) & (counter[0] < np.percentile(counter[0], 50))] = 3
        counter[7][(counter[0] >= np.percentile(counter[0], 50)) & (counter[0] < np.percentile(counter[0], 70))] = 4
        counter[7][(counter[0] >= np.percentile(counter[0], 70)) & (counter[0] <= np.percentile(counter[0], 90))] = 5
        counter[7][counter[0] > np.percentile(counter[0], 90)] = 7


    # remove stragety

        th0 = np.percentile(counter[0], 2)
        removed0 = np.where(counter[0] < th0)[0]

        th1 = np.percentile(counter[1], 2)
        removed1 = np.where(counter[1] < th1)[0]

        th2 = np.percentile(counter[2], 2)
        removed2 = np.where(counter[2] < th2)[0]

        th3 = np.percentile(counter[3], 2)
        removed3 = np.where(counter[3] < th3)[0]

        th4 = np.percentile(counter[4], 2)
        removed4 = np.where(counter[4] < th4)[0]

        th5 = np.percentile(counter[5], 2)
        removed5 = np.where(counter[5] < th5)[0]

        # counter 6
        th6 = np.percentile(counter[6],2)
        removed6 = np.where(counter[6] < th6)[0]

        th7 = np.percentile(counter[6], 5)
        removed7 = np.where(counter[6] < th7)[0]

        th8 = np.percentile(counter[6],95)
        removed8 = np.where(counter[6]>th8)[0]

        th9 = np.percentile(counter[6], 98)
        removed9 = np.where(counter[6] > th9)[0]

        # counter 7
        th10 = np.percentile(counter[7], 2)
        removed10 = np.where(counter[7] < th10)[0]

        th11 = np.percentile(counter[7], 5)
        removed11 = np.where(counter[7] < th11)[0]

        th12 = np.percentile(counter[7], 95)
        removed12 = np.where(counter[7] > th12)[0]

        th13 = np.percentile(counter[7], 98)
        removed13 = np.where(counter[7] > th13)[0]

        infl_src = InflSet(domain_adaptation_task, 'source', src_repetition, sample_per_class)

        print("infl_src : ", len(infl_src))

        removed_random_5 = np.random.permutation(np.arange(len(infl_src)))[:int(len(infl_src)*(5 * 0.01))]

        removed_random_2 = np.random.permutation(np.arange(len(infl_src)))[:int(len(infl_src) * (2 * 0.01))]

        removed_random_5e = np.random.permutation(np.arange(len(infl_src)))[:5]

    # sample rate stragety: use counter[i] as weight

        sample_weight = {}
        sample_weight['pure_sum'] = (counter[0] + abs(np.amin(counter[0])) + 0.1).tolist()
        sample_weight['tri_2std'] = (counter[1] + abs(np.amin(counter[1])) + 0.1).tolist()
        # sample_weight['bi_neg_1std'] = (counter[2] + abs(np.amin(counter[2])) + 0.1).tolist()
        # sample_weight['bi_neg_2std'] = (counter[3] + abs(np.amin(counter[3])) + 0.1).tolist()
        # sample_weight['bi_pos_1std'] = (counter[4] + abs(np.amin(counter[4])) + 0.1).tolist()
        # sample_weight['abs_2std'] = (counter[5] + abs(np.amin(counter[5])) + 0.1).tolist()
        sample_weight['4_seg'] = counter[6].tolist()
        sample_weight['6_seg'] = counter[7].tolist()
        # sample_weight['random_weight1'] = np.random.permutation(counter[0] + abs(np.amin(counter[0])) + 0.1).tolist()
        sample_weight['random_weight2'] = np.random.rand(counter[0].shape[0]).tolist()

        sample_weight_path = config['outdir']+f'/sample_weight_{domain_adaptation_task}'\
                        f'_s{str(sample_per_class)}_r{str(repetition)}.json'
        save_json(sample_weight, sample_weight_path)


        stats = {}
        # stats['pure_sum'] = removed0.tolist()
        stats['tri_2std'] = removed1.tolist()
        # stats['bi_neg_1std'] = removed2.tolist()
        # stats['bi_neg_2std'] = removed3.tolist()
        # stats['bi_pos_1std'] = removed4.tolist()
        # stats['abs_2std'] = removed5.tolist()
        # stats['4_seg_2per'] = removed6.tolist()
        # stats['4_seg_5per'] = removed7.tolist()
        # stats['4_seg_95per'] = removed8.tolist()
        # stats['4_seg_98per'] = removed9.tolist()
        # stats['6_seg_2per'] = removed10.tolist()
        # stats['6_seg_5per'] = removed11.tolist()
        # stats['6_seg_95per'] = removed12.tolist()
        # stats['6_seg_98per'] = removed13.tolist()
        stats['random_reomove_2per'] = removed_random_2.tolist()
        # stats['random_reomove_5per'] = removed_random_5.tolist()
        # stats['random_remove_5example'] = removed_random_5e.tolist()
        time = dt.now().strftime("%Y-%m-%d-%H-%M-%S")
        stats_path = config['outdir']+f'/infl_std_stats_{domain_adaptation_task}'\
                        f'_s{str(sample_per_class)}_r{str(repetition)}.json'
        save_json(stats, stats_path)
    
    # if mode in ['all', 'retrain']:
        # stats_path = config['outdir']+f'infl_stats_{domain_adaptation_task}'\
        #                 f'_s{str(sample_per_class)}_r{str(repetition)}.json'

        with open(sample_weight_path) as json_file:
            sample_weight = json.load(json_file)

        with open(stats_path) as json_file:
            data = json.load(json_file)

        device = torch.device("cuda")
        net = Network().to(device)

    # remove indices
        results[str(repetition)]['retrain_remove_indices'] = {}
        for l in data:
            results[str(repetition)]['retrain_remove_indices'][l] = {}
            removed_indices = data[l]
            retrain_set = TrainSet(domain_adaptation_task, 'baseline', src_repetition, tgt_repetition, sample_per_class, removed_indices)
            retrain_set_loader = DataLoader(retrain_set, batch_size=batch, shuffle=True)
            excel = []
            sum_re_acc = 0
            for i in range(10):

                net = Network().to(device)
                best_test_acc = 0
                for epoch in range(epochs):
                    train_loss = train(device, net, retrain_set_loader)
                    test_acc = test(device, net, test_set_loader)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        save_model(net, NEW_PATH)
                results[str(repetition)]['retrain_remove_indices'][l][i] = best_test_acc
                excel.append(best_test_acc)
                print(l +' at time ' + str(i) + ' has retraining best test acc :', best_test_acc)
                sum_re_acc += best_test_acc
            results[str(repetition)]['retrain_remove_indices'][l]['avg_re_acc'] = sum_re_acc/10
            print(l + ' has average retrain test acc :', sum_re_acc/10)
            excel.append(sum_re_acc/10)
            for i in excel:
                print(i)

    # sample weight
        results[str(repetition)]['retrain_sample_weight'] = {}
        for w in sample_weight:
            results[str(repetition)]['retrain_sample_weight'][w] = {}
            m = max(sample_weight[w]) * 1.2
            weight = sample_weight[w] + [m]*10*sample_per_class
            sampler = WeightedRandomSampler(weight, len(sample_weight[w]),replacement=True)
            retrain_set = TrainSet(domain_adaptation_task, 'baseline', src_repetition, tgt_repetition, sample_per_class)
            retrain_set_loader = DataLoader(retrain_set, batch_size=batch, shuffle=False, sampler=sampler)

            sum_re_acc = 0
            excel = []
            for i in range(10):
                net = Network().to(device)
                best_test_acc = 0
                for epoch in range(epochs):
                    train_loss = train(device, net, retrain_set_loader)
                    test_acc = test(device, net, test_set_loader)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        # save_model(net, NEW_PATH)
                results[str(repetition)]['retrain_sample_weight'][w][i] = best_test_acc
                print(w + ' at time ' + str(i) + ' has retraining best test acc :', best_test_acc)
                excel.append(best_test_acc)
                sum_re_acc += best_test_acc
            results[str(repetition)]['retrain_sample_weight'][w]['avg_re_acc'] = sum_re_acc / 10
            print(w + ' average retrain test acc :', sum_re_acc / 10)
            excel.append(sum_re_acc / 10)
            for i in excel:
                print(i)

    # sample weight + cssa
        results[str(repetition)]['retrain_sample_weight_CSSA'] = {}
        for w in sample_weight:
            results[str(repetition)]['retrain_sample_weight_CSSA'][w] = {}
            weight = sample_weight[w]
            retrain_set = TrainSet_CSSA(domain_adaptation_task,src_repetition, tgt_repetition, sample_per_class, weights=weight)
            sample_rate = retrain_set.weights
            sampler = WeightedRandomSampler(sample_rate, len(sample_weight[w]), replacement=True)
            retrain_set_loader = DataLoader(retrain_set, batch_size=batch, shuffle=False, sampler=sampler)
            # retrain_set_loader = DataLoader(retrain_set, batch_size=batch, shuffle=True)


            sum_re_acc = 0
            excel = []
            for i in range(10):
                net = Network().to(device)
                best_test_acc = 0
                for epoch in range(epochs):
                    train_loss = train_CSSA(device, net, retrain_set_loader,'CSSA')
                    test_acc = test(device, net, test_set_loader)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        # save_model(net, NEW_PATH)
                results[str(repetition)]['retrain_sample_weight_CSSA'][w][i] = best_test_acc
                print(w + ' at time ' + str(i) + ' has retraining best test acc :', best_test_acc)
                excel.append(best_test_acc)
                sum_re_acc += best_test_acc
            results[str(repetition)]['retrain_sample_weight_CSSA'][w]['avg_re_acc'] = sum_re_acc / 10
            print(w + ' average retrain test acc :', sum_re_acc / 10)
            excel.append(sum_re_acc / 10)
            for i in excel:
                print(i)


        print("\n plain cssa \n")
    # plain cssa
        results[str(repetition)]['retrain_CSSA'] = {}
        for w in sample_weight:
            results[str(repetition)]['retrain_CSSA'][w] = {}
            weight = sample_weight[w]
            retrain_set = TrainSet_CSSA(domain_adaptation_task, src_repetition, tgt_repetition, sample_per_class,
                                        weights=weight)
            # sample_rate = retrain_set.weights
            # sampler = WeightedRandomSampler(sample_rate, len(sample_weight[w]), replacement=True)
            # retrain_set_loader = DataLoader(retrain_set, batch_size=batch, shuffle=False, sampler=sampler)
            retrain_set_loader = DataLoader(retrain_set, batch_size=batch, shuffle=True)

            sum_re_acc = 0
            excel = []
            for i in range(10):
                net = Network().to(device)
                best_test_acc = 0
                for epoch in range(epochs):
                    train_loss = train_CSSA(device, net, retrain_set_loader, 'CSSA')
                    test_acc = test(device, net, test_set_loader)
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        # save_model(net, NEW_PATH)
                results[str(repetition)]['retrain_CSSA'][w][i] = best_test_acc
                print(w + ' at time ' + str(i) + ' has retraining best test acc :', best_test_acc)
                excel.append(best_test_acc)
                sum_re_acc += best_test_acc
            results[str(repetition)]['retrain_CSSA'][w]['avg_re_acc'] = sum_re_acc / 10
            print(w + ' average retrain test acc :', sum_re_acc / 10)
            excel.append(sum_re_acc / 10)
            for i in excel:
                print(i)

            break

    return results


    # if mode == 'all':
    #     with open(logfile, 'a+') as f_out:
    #         f_out.write('-'*50+'\n')
    #         f_out.write('influence cutoff:\n')
    #         for per in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
    #             f_out.write(str(per)+'% :'+str(np.percentile(cumulative_influence, per))+'\n')
    #         f_out.write('dataset : '+domain_adaptation_task+'\n')
    #         f_out.write('sample_per_class : '+str(sample_per_class)+'\n')
    #         f_out.write('repetition : '+str(repetition)+'\n')
    #         f_out.write('method : '+method+'\n')
    #         f_out.write('sampling : '+sampling+'\n')
    #         f_out.write('remove_perc : '+str(remove_perc)+'\n')
    #         f_out.write('batch :' +str(batch)+'\n')
    #         f_out.write('epochs : '+str(epochs)+'\n')
    #         f_out.write('retrain_epochs : '+str(retrain_epochs)+'\n')
    #         f_out.write('alpha : '+str(alpha)+'\n')
    #         f_out.write(results[0][0]+' : '+str(results[0][1])+'\n')
    #         for i in range(1, len(results)):
    #             f_out.write(results[i][0]+' with '+str(results[i][2])+'% removed'+' : '+str(results[i][1])+'\n')
    #         f_out.write('-'*50+'\n')

if __name__ == '__main__':
    """
    mode: chosen from ['all', 'train', 'influence', 'retrain']
    domain_adaptation_task: chosen from ['USPS_to_MNIST', 'MNIST_to_USPS']
    sample_per_class: 1-9
    repetition: 1-7
    weight_strategies: a list of subset of ['remove_most_beneficial', 'remove_most_harmful', 'remove_most_influential', 'remove_no_influential', 'step']
    sampling: chosen from ['plain', 'pair']
    'plain': source training and target training are concatenated in Dataset
    train_plain will be used.
    'pair': source training and target training are paired in Dataset.
    train will be used
    if 'plain', method will be ignored and can only be source_target
    method: chosen from ['source', 'source_target', 'CSSA']
    remove_percs: a list of the percentage of source data to be removed. it won't be used if weight_strategy = 'step'
    batch:
    epochs:
    retrain_epochs:
    alpha: it will be used only when method = 'CSSA'
    logfile: file to record results

    mode = 'train'
    """
    mode = 'all'
    domain_adaptation_task = 'USPS_to_MNIST'
    sample_per_class = 1
    src_repetition = [[4]]
    tgt_repetition = [[4]]
    batch = 128
    epochs = 320
    retrain_epochs = 200
    alpha = 0.25
    logfile = 'log.txt'
    config = get_default_config()
    results = {}

    for i in range(1):
        results = run_trial(results, mode, domain_adaptation_task, sample_per_class, src_repetition[i], tgt_repetition[i], batch, epochs, retrain_epochs, alpha)

    results_path = config['outdir']+f'/results_{domain_adaptation_task}'\
                    f'_s{str(sample_per_class)}_r{str((src_repetition, tgt_repetition))}.json'
    save_json(results, results_path)
    # cnt = 0
    # # for domain_adaptation_task in ['USPS_to_MNIST', 'MNIST_to_USPS']:
    # for domain_adaptation_task in ['MNIST_to_USPS']:
    #     for sample_per_class in range(1, 3):
    #         for repetition in range(1, 3):
    #             cnt += 1
    #             print(cnt)
    #             weight_strategies = ['remove_most_beneficial', 'remove_most_harmful', 'remove_most_influential', 'remove_no_influential']
    #             remove_percs = [5, 20, 40]
    #             run_trial(mode, domain_adaptation_task, sample_per_class, repetition, weight_strategies, method, sampling, remove_percs, batch, epochs, retrain_epochs, alpha, logfile)