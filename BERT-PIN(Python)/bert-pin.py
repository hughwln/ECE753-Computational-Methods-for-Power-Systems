import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
import model

class Profile_Dataset(Dataset):
    def __init__(self, data, dim_input):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = data

        self.dim_input = dim_input
        self.dataset = torch.from_numpy(self.dataset).float().to(self.device)

    def __getitem__(self, index):
        # load_patched, mask, T, load_gt
        model_input = self.dataset[index, 0:self.dim_input * 1].reshape([1, self.dim_input])
        temperature = self.dataset[index, self.dim_input * 2:self.dim_input * 3].reshape([1, self.dim_input])
        mask = self.dataset[index, self.dim_input * 1:self.dim_input * 2].bool().reshape([1, self.dim_input])

        return model_input, temperature, mask

    def __len__(self):
        return len(self.dataset)

def showmaskeddata():
    raw_data = pd.read_csv('../data/load_temperature_masked.csv', index_col=0)
    raw_data.index = pd.to_datetime(raw_data.index)
    plt.plot(raw_data.loc['2018-3-18', 'load'].to_numpy(), label='masked')
    plt.show()

def generate_data():
    raw_data = pd.read_csv('../data/load_temperature_data.csv', index_col=0)
    raw_data.index = pd.to_datetime(raw_data.index)
    real_loads = raw_data.iloc[:, 0].to_numpy().copy()
    real_T = raw_data.iloc[:, -1].to_numpy()
    pmax = np.amax(real_loads)
    pmin = np.amin(real_loads)
    tmax = np.amax(real_T)

    d = '18'
    dates = ['2018-3-' + d, '2018-5-' + d, '2018-7-' + d, '2018-9-' + d, '2018-11-' + d]
    gt_list = []
    for day in dates:
        df = raw_data.loc[day+' 10:00:00': day+' 13:45:00', 'load'].copy()
        gt_list.append(df)
        raw_data.loc[day + ' 10:00:00': day + ' 13:45:00', 'load'] = 0
    gt = pd.concat(gt_list)
    gt.to_csv('gt.csv', index=False, header=False)
    raw_data.to_csv('load_temperature_masked.csv')
    plt.plot(real_loads, label="raw")
    plt.plot(raw_data.loc[:, 'load'].to_numpy(), label='masked')
    plt.legend()
    plt.show()

def read_data():
    raw_data = pd.read_csv('../data/load_temperature_masked.csv', index_col=0)
    raw_data.index = pd.to_datetime(raw_data.index)
    real_loads = raw_data.iloc[:, 0].to_numpy().copy()
    real_T = raw_data.iloc[:, -1].to_numpy().copy()
    pmax = np.amax(real_loads)
    pmin = np.amin(real_loads)
    tmax = np.amax(real_T)
    raw_data.iloc[:, 0] = raw_data.iloc[:, 0] / np.amax(real_loads)
    raw_data.iloc[:, 1] = raw_data.iloc[:, 1] / np.amax(real_T)

    d = '18'
    dates = ['2018-3-' + d, '2018-5-' + d, '2018-7-' + d, '2018-9-' + d, '2018-11-' + d]
    norm_set = np.zeros([5, 96 * 3])  # input/mask/temperature
    pt_norm = 0
    for day in dates:
        load_patched = raw_data.loc[day, 'load'].to_numpy()
        T = raw_data.loc[day, 'temperature'].to_numpy()
        patch_start = 96 // 2 - 16 // 2
        mask = np.zeros(96, dtype=bool)
        mask[patch_start:patch_start + 16] = True
        norm_set[pt_norm, :] = np.concatenate((load_patched, mask, T), axis=0, dtype=float).reshape(1, 96 * 3)
        pt_norm += 1

    return norm_set, pmax, tmax

def run_bert_pin():
    # gt = pd.read_csv('../data/gt.csv', header=None)
    # gt_np = gt.to_numpy()

    norm_set, pmax, tmax = read_data()
    # gt_np = gt_np / pmax
    # gt_np = gt_np.reshape((5, 16))
    testset = Profile_Dataset(data=norm_set, dim_input=96)
    testloader = DataLoader(testset, batch_size=5, num_workers=0, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 202
    heads = 2
    transformer = model.Transformer(src_vocab_size=K + heads, embed_size=K + heads, heads=heads, num_layers=2,
                                    forward_expansion=heads, max_length=96, device=device).to(device)
    transformer.load_state_dict(torch.load('encoder_100.pth'))
    output_list = []

    for i, data in enumerate(testloader):
        model_input, temperature, mask = data
        model_input = model_input.reshape((model_input.shape[0], model_input.shape[2])).to(device)
        mask = mask.reshape((mask.shape[0], mask.shape[2])).to(device)
        temperature = temperature.reshape((temperature.shape[0], temperature.shape[2])).to(device)
        mask_np = mask.cpu().detach().numpy()

        # BERT
        bert_input = model_input.clone()
        bert_input = torch.round(bert_input * K).type(torch.int)
        temperature = torch.round(temperature * K).type(torch.int)
        bert_output = transformer(bert_input, temperature, mask)
        blin_output = bert_output.argmax(dim=-1)
        blin_output = blin_output / K
        blin_output = model_input[:, :] + blin_output * mask

        pre_np_bert = blin_output.cpu().detach().numpy()
        output_np = pre_np_bert[:, 4*10: 4*14].reshape((-1, 1))
        output_list.append(output_np)
        input_np = model_input.cpu().detach().numpy()

        fig = plt.figure(1, figsize=(2 * 4, 2 * 2))
        plt.clf()
        gs = fig.add_gridspec(2, 3)
        for j in range(5):
            patch_bgn = np.where(mask_np[j, :] == 1.0)[0][0] - 1
            patch_end = np.where(mask_np[j, :] == 1.0)[0][-1] + 1
            x = np.arange(patch_bgn, patch_end + 1)
            y_min = np.amin(pre_np_bert)
            y_max = np.amax(pre_np_bert)

            ax = fig.add_subplot(gs[j // 3, j % 3])
            # draw input data
            ax.plot(np.arange(0, patch_bgn + 1), input_np[j, 0:patch_bgn + 1], 'k', linewidth=1)
            ax.plot(np.arange(patch_end, np.size(input_np, axis=1)), input_np[j, patch_end:], 'k', linewidth=1)

            # draw output data
            ax.plot(x, pre_np_bert[j, patch_bgn:patch_end + 1], 'r', linewidth=1, label='BERT-PIN')
            # ax.plot(np.arange(40, 56), gt_np[j], 'g', linewidth=1, label="GT")

            plt.ylim(y_min, y_max)
            plt.xticks([])
            plt.yticks([])
            rect = plt.Rectangle((patch_bgn, y_min), patch_end - patch_bgn + 1, y_max - y_min,
                                 facecolor="k", alpha=0.1)
            ax.add_patch(rect)
            ax.legend()

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

    output_csv = np.concatenate(output_list)
    output_csv = output_csv * pmax
    df = pd.DataFrame(output_csv)
    df.to_csv('output.csv', index=False, header=False)

if __name__ == '__main__':
    # generate_data()
    run_bert_pin()

    # showmaskeddata()