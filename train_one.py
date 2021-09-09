import torch


def train_one_epoch(model, data_loader, device, epoch, writer):
    start = 1419

    for i, (images, lidar, targets, cal) in enumerate(data_loader):
        if i < 1000:
            continue
        # print('error?!!!!!!!!!!!!!!!!!')
        print("@@@@@@@@@[Epoch] : ", i)

        # model.train()
        num_iters = epoch * len(data_loader) + i
        # images = images[0].to(device)
        images = images[0]
        lidar = lidar[0]
        cal = cal[0]

        targets = {k: v.to(device) for k, v in targets[0].items()}
        model(images, lidar, targets, cal, device)
        
     

def evaluate(mode, data_loader, device):
    dataset = data_loader 