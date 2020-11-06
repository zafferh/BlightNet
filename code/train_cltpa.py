import torchvision.transforms as transforms
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from config import *
from dataset import BlightDataset
from model_ltpa import AttnVGG_before, AttnVGG_after

import matplotlib.pyplot as plt
from PIL import Image
from utilities import *

def _init_fn(worker_id):
    np.random.seed(12+worker_id)

if __name__ == '__main__':
    H, W = 256, 256
    for m in range(len(WINDOW_SIZES)):
        hh, ww = WINDOW_SIZES[m]
        stride = STRIDES[m]
        for i in range(0, H - hh + 1, stride):
            for j in range(0, W - ww + 1, stride):

                transform_train = transforms.Compose([
                    transforms.Resize((IM_SIZE, IM_SIZE), interpolation=Image.NEAREST),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.36534598, 0.36992216, 0.33446208), (0.2657754, 0.26731136, 0.2778652))
                ])
                # transformations to apply for test. ToTensor and normalization.
                transform_test = transforms.Compose([
                    transforms.Resize((IM_SIZE, IM_SIZE), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                    transforms.Normalize((0.36534598, 0.36992216, 0.33446208), (0.2657754, 0.26731136, 0.2778652))
                ])
                cropping = (j, i, j+ww, i+hh)
                print('Initializing network with cropping ({:d}, {:d}, {:d}, {:d}) and window size of {:d}.'
                      .format(j, i, j+ww, i+hh, hh))
                trainset = BlightDataset(root='data', set_type='train', transform=transform_train, cropping=cropping)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE_CLTPA, shuffle=True,
                                                          num_workers=8, worker_init_fn = _init_fn)
                testset = BlightDataset(root='data', set_type='val', transform=transform_test, cropping=cropping)
                testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=5)

                if ATTN_MODE_CLTPA == 'before':
                    print('\npay attention before maxpooling layers...\n')
                    net = AttnVGG_before(im_size=IM_SIZE, num_classes=2,
                        attention = True, normalize_attn = True, init = 'xavierUniform', interpolate=True)
                elif ATTN_MODE_CLTPA == 'after':
                    print('\npay attention after maxpooling layers...\n')
                    net = AttnVGG_after(im_size=IM_SIZE, num_classes=2,
                        attention = True, normalize_attn = True, init = 'xavierUniform', interpolate=True)
                else:
                    raise NotImplementedError("Invalid attention mode!")

                criterion = nn.CrossEntropyLoss()
                print('done')

                ## move to GPU
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                device_ids = [0]
                model = nn.DataParallel(net, device_ids=device_ids).to(device)
                criterion.to(device)

                ### optimizer

                # momentum of 0.9, weight decay of 5e-4,
                optimizer = optim.SGD(model.parameters(), lr=LR_CLTPA, momentum=0.9, weight_decay=5e-4)
                # The learning rate is scaled by 0.5 every 25 epochs
                lr_lambda = lambda epoch: np.power(0.5, int(epoch / 25))
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

                # training
                step = 0
                running_avg_accuracy = 0
                best_test_accuracy = 0
                num_aug = 1
                for epoch in range(EPOCHS_CLTPA):
                    images_disp = []
                    # adjust learning rate
                    scheduler.step()
                    print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
                    # run for one epoch
                    for aug in range(num_aug):
                        for k, data in enumerate(trainloader, 0):
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
                            if k % 10 == 0:
                                model.eval()
                                pred, __, __, __ = model(inputs)
                                predict = torch.argmax(pred, 1)
                                total = labels.size(0)
                                correct = torch.eq(predict, labels).sum().double().item()
                                accuracy = correct / total
                                running_avg_accuracy = 0.9 * running_avg_accuracy + 0.1 * accuracy
                                print(
                                    "[epoch %d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                                    % (epoch, k, len(trainloader) - 1, loss.item(), (100 * accuracy),
                                       (100 * running_avg_accuracy)))
                            step += 1

                    # the end of each epoch: test & log
                    print('\none epoch done, saving records ...\n')

                    model.eval()
                    total = 0
                    correct = 0
                    with torch.no_grad():
                        # log scalars
                        for k, data in enumerate(testloader, 0):
                            images_test, labels_test = data
                            images_test, labels_test = images_test.to(device), labels_test.to(device)
                            pred_test, __, __, __ = model(images_test)
                            predict = torch.argmax(pred_test, 1)
                            total += labels_test.size(0)
                            correct += torch.eq(predict, labels_test).sum().double().item()
                        print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100 * correct / total))
                        if correct / total > best_test_accuracy:
                            print(
                                'Saving new best model... Previous best model had accuracy of %.4f\n' % best_test_accuracy)
                            best_test_accuracy = correct / total
                            torch.save(model.state_dict(),
                                       os.path.join(LOG_DIR, 'net_{:d}_{:.2f}_{:d}_{:d}_{:d}.pth'
                                                    .format(epoch, best_test_accuracy, i, j, hh)))
                print('Saving final model. It has accuracy of %.4f\n' % (correct / total))
                torch.save(model.state_dict(),
                           os.path.join(LOG_DIR, 'net_{:d}_{:.2f}_{:d}_{:d}_{:d}.pth'
                                        .format(epoch, correct / total, i, j, hh)))
