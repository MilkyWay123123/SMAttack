'''
Manifold attack
'''
import torch.nn
from torch.utils.data import DataLoader

from utils.loadModel import *
from utils.ThinPlateSpline import *
from feeder.feeder import *

import argparse
from omegaconf import OmegaConf


def foolRateCal(args, rlabels, flabels, logits=None):
    hitIndices = []

    if args.attackType == 'ab':
        for i in range(0, len(flabels)):
            if flabels[i] != rlabels[i]:
                hitIndices.append(i)
    else:
        print('Error, no implemented')
        return

    return len(hitIndices) / len(flabels) * 100


def getRA(data_freq, Len=300):
    F_single_sided = (2 / Len) * data_freq
    R = torch.abs(F_single_sided)
    A = torch.angle(F_single_sided)

    return R, A


def complex_freq(R, A, Len=300):
    F_prime = R * torch.exp(1j * A)

    F_prime = F_prime * Len / 2

    return F_prime

def project_space_time(data_freq):
    OriginData = torch.fft.ifft(data_freq, dim=2).real

    return OriginData


# 固定采样每帧的节点
Vs = [0, 1, 2, 3,
      7, 11, 15, 19,
      9, 13, 17, 21]

def get_sample_points(data):
    N, C, T, V, M = data.shape
    points = torch.zeros([N, C, T, len(Vs), M])
    for i in range(len(Vs)):
        points[:, :, :, i, :] = data[:, :, :, Vs[i], :]
    points = points.permute(0, 2, 4, 3, 1).contiguous().view(N * T * M, len(Vs), C)
    return points


def rand_data(subsidiary_dataset, index):
    for batchNo, (data) in enumerate(subsidiary_dataset):
        if batchNo == index:
            return data


def unspecificAttack(labels, classNum=60):
    flabels = np.ones((len(labels), classNum))
    flabels = flabels * 1 / classNum

    return torch.LongTensor(flabels)


def attack(args):
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)

    frame_len = 300
    feeder = Feeder(args.data_path, args.label_path)
    trainloader = DataLoader(feeder,
                             batch_size=args.batch_size,
                             num_workers=4, pin_memory=True, drop_last=True, shuffle=False)

    subsidiary_dataset = DataLoader(Feeder2(args.sub_path),
                                    batch_size=args.batch_size,
                                    num_workers=4, pin_memory=True, drop_last=True, shuffle=True)

    epochs = args.epochs
    size = len(subsidiary_dataset)
    overall_raio = 0
    deltaT = 1 / 30

    model = getModel(args.model_name)

    SourceSample = []
    AttackSample = []
    LabelSample = []

    for batchNo, (data, label) in enumerate(trainloader):
        data = data.cuda()
        label = label.cuda()
        N, C, T, V, M = data.shape

        adData = data.clone()
        adData.requires_grad = True
        trajectory_freq = torch.fft.fft(adData, dim=2)
        R, A = getRA(trajectory_freq, frame_len)
        R_max = torch.max(R)
        S_r = (R / R_max)

        # Two samples are randomly selected to generate a random amplitude map
        template_s = rand_data(subsidiary_dataset, np.random.randint(0, size - 1)).cuda()
        trajectory_s_freq = torch.fft.fft(template_s, dim=2)
        R_s, A_s = getRA(trajectory_s_freq, frame_len)

        template_r = rand_data(subsidiary_dataset, np.random.randint(0, size - 1)).cuda()
        trajectory_r_freq = torch.fft.fft(template_r, dim=2)
        R_r, A_r = getRA(trajectory_r_freq, frame_len)

        weights = (S_r * (R_s - R_r)).contiguous().clone().detach().cuda()
        adWeights = weights.clone()
        adWeights.requires_grad = True

        source_space_points = get_sample_points(adData)

        learning_rate = args.lr
        for epoch in range(epochs):
            # For the adversarial part, a new sample is calculated each time
            R_p = R + adWeights
            R_p.data = torch.clamp(R_p.data, 0, torch.max(R_p).item())
            R_diff = (R - R_p).sum()
            proData = project_space_time(complex_freq(R_p, A, frame_len))

            # The constraint on the spine
            spine_points = [0, 1, 20]
            proData.data[:, :, :, spine_points, :] = data.data[:, :, :, spine_points, :]

            target_points = get_sample_points(proData)
            tps = ThinPlateSpline(source_space_points, target_points)
            tempData = adData.permute(0, 2, 4, 3, 1).contiguous().view(N * T * M, V, C)
            tempData = tps.transform(tempData)
            tempData = tempData.view(N, T, M, V, C).permute(0, 4, 1, 3, 2)
            tempData = tempData.cuda()

            pred = model(tempData)
            predictedLabels = torch.argmax(pred, dim=1)

            acc_loss = (tempData[:, :, 2:, Vs, :] - 2 * tempData[:, :, 1:-1, Vs, :] + tempData[:, :, :-2, Vs,
                                                                                      :]) / deltaT / deltaT
            acc_loss = torch.sqrt(torch.mean(acc_loss ** 2))
            classify_loss = -0.05 * torch.nn.CrossEntropyLoss()(pred, label) + acc_loss

            loss = classify_loss

            adWeights.grad = None
            loss.backward(retain_graph=True)
            cgs = adWeights.grad

            cgsView = cgs.view(cgs.shape[0], -1)
            cgsnorms = torch.norm(cgsView, dim=1) + 1e-18
            cgsView /= cgsnorms[:, np.newaxis]

            with torch.no_grad():
                missedIndices = []
                if args.attackType == 'ab':
                    for i in range(len(label)):
                        if label[i] == predictedLabels[i]:
                            missedIndices.append(i)
                else:
                    print('Error, no implemented')
                    return

                adWeights[missedIndices] = adWeights[missedIndices] - cgs[missedIndices].sign() * learning_rate

            if args.attackType == 'ab':
                foolRate = foolRateCal(args, label, predictedLabels)
            else:
                print('Error, no implemented')
                return

            if epoch % 20 == 0:
                print(
                    f'Epoch:{epoch} loss={loss} classify_loss={classify_loss} foolRate={foolRate} learning_rate={learning_rate}')

            if foolRate >= 100 or epoch == epochs - 1:
                overall_raio += foolRate
                print(
                    f'Manifold attack was successful,batchNo={batchNo} epoch:{epoch} overall_raio={overall_raio / (batchNo + 1)} Current round success rate={foolRate}')
                for i in range(len(label)):
                    if label[i] != predictedLabels[i]:
                        SourceSample.append(data[i].detach().clone().cpu())
                        AttackSample.append(tempData[i].detach().clone().cpu())
                        LabelSample.append(label[i].detach().clone().cpu())

                break

        if len(SourceSample) != 0:
            # 保存
            np.save(f'{args.save_path}/SourceSample.npy', np.stack(SourceSample))
            np.save(f'{args.save_path}/AttackSample.npy', np.stack(AttackSample))
            np.save(f'{args.save_path}/LabelSample.npy', np.stack(LabelSample))


if __name__ == "__main__":
    print('SMAttack')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    print(config)
    attack(config)
