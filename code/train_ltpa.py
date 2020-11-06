import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision.transforms as transforms

from config import *
from model_ltpa import AttnVGG_before, AttnVGG_after
from utilities import *
from dataset import BlightDataset
from PIL import Image

def _init_fn(worker_id):
    np.random.seed(12+worker_id)

def main():
    print('\nloading the dataset ...\n')
    num_aug = 1
    print('IM_SIZE = {}'.format(IM_SIZE))
    #transformations to apply for training. Includes some data aaugmentations operations
    # Cifar is already 32x32 dataset, so the RandomCrop is just for random shifts.
    transform_train = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=Image.NEAREST),
        transforms.RandomCrop(IM_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        # 0-255 to 0-1 range
        transforms.ToTensor(),
        transforms.Normalize((0.36534598,0.36992216,0.33446208), (0.2657754 , 0.26731136, 0.2778652))
    ])
    # transformations to apply for test. ToTensor and normalization.
    transform_test = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.36534598,0.36992216,0.33446208), (0.2657754 , 0.26731136, 0.2778652))
    ])

    trainset = BlightDataset(root='data', set_type='train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE_LTPA, shuffle=True, num_workers=8, worker_init_fn=_init_fn)
    testset = BlightDataset(root='data', set_type='val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=5)

    if ATTN_MODE_LTPA == 'before':
        print('\npay attention before maxpooling layers...\n')
        net = AttnVGG_before(im_size=IM_SIZE, num_classes=2,
            attention=True, normalize_attn=True, init='xavierUniform')
    elif ATTN_MODE_LTPA == 'after':
        print('\npay attention after maxpooling layers...\n')
        net = AttnVGG_after(im_size=IM_SIZE, num_classes=2,
            attention=True, normalize_attn=True, init='xavierUniform')
    else:
        raise NotImplementedError("Invalid attention mode!")

    '''
    class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')[source]
This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

It is useful when training a classification problem with C classes. 
    '''
    criterion = nn.CrossEntropyLoss()
    print('done')


    ## move to GPU
    print('\nmoving to GPU ...\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    print('done')

    ### optimizer

    # momentum of 0.9, weight decay of 5e-4,
    optimizer = optim.SGD(model.parameters(), lr=LR_LTPA, momentum=0.9, weight_decay=5e-4)
    # The learning rate is scaled by 0.5 every 25 epochs
    lr_lambda = lambda epoch : np.power(0.5, int(epoch/25))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    print('\nstart training ...\n')
    step = 0
    running_avg_accuracy = 0
    best_test_accuracy = 0
    for epoch in range(EPOCHS_LTPA):
        images_disp = []
        # adjust learning rate
        scheduler.step()
        print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
        # run for one epoch
        for aug in range(num_aug):
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                pred, __, __, __ = model(inputs)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    model.eval()
                    pred, __, __, __ = model(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    running_avg_accuracy = 0.9*running_avg_accuracy + 0.1*accuracy
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                        % (epoch, aug, num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*running_avg_accuracy)))
                step += 1

        # the end of each epoch: test & log
        print('\none epoch done, saving records ...\n')

        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            # log scalars
            for i, data in enumerate(testloader, 0):
                images_test, labels_test = data
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                pred_test, __, __, __ = model(images_test)
                predict = torch.argmax(pred_test, 1)
                total += labels_test.size(0)
                correct += torch.eq(predict, labels_test).sum().double().item()
            print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100*correct/total))
            if correct/total > best_test_accuracy:
                print('Saving new best model... Previous best model had accuracy of %.4f\n' % best_test_accuracy)
                best_test_accuracy = correct/total
                torch.save(model.state_dict(), os.path.join(LOG_DIR, 'net_{:d}_{:.2f}.pth'.format(epoch, best_test_accuracy)))
    print('Saving final model. It has accuracy of %.4f\n' % (correct/total))
    torch.save(model.state_dict(), os.path.join(LOG_DIR, 'net_{}_{:d}_{:.2f}.pth'
                                                .format(ATTN_MODE_LTPA, epoch, correct/total)))

if __name__ == "__main__":
    main()
