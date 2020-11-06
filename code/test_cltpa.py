from config import *
from dataset import BlightDataset
from PIL import Image
from model_ltpa import AttnVGG_before, AttnVGG_after
import torchvision.transforms as transforms
from utilities import *
import os
from imutils import paths

if __name__ == '__main__':
    H, W = 256, 256
    count = 0
    models=[]
    base_dir = 'data'
    base_test_dir = os.path.join(base_dir, 'test')

    total_test = len(list(paths.list_images(base_test_dir)))
    print('Evaluating test set of size {:d}.'.format(total_test))
    preds_sum = np.zeros((total_test, 2))
    for m in range(len(WINDOW_SIZES)):
        hh, ww = WINDOW_SIZES[m]
        stride = STRIDES[m]
        for i in range(0, H - hh + 1, stride):
            for j in range(0, W - ww + 1, stride):
                transform_test = transforms.Compose([
                    transforms.Resize((IM_SIZE, IM_SIZE), interpolation=Image.NEAREST),
                    transforms.ToTensor(),
                    transforms.Normalize((0.36534598, 0.36992216, 0.33446208), (0.2657754, 0.26731136, 0.2778652))
                ])
                cropping = (j, i, j+ww, i+hh)
                print('Initializing network with cropping ({:d}, {:d}, {:d}, {:d}) and window size of {:d}.'
                      .format(j, i, j+ww, i+hh, hh))

                testset = BlightDataset(root='data', set_type='test', transform=transform_test, cropping=cropping)
                testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=5)

                if ATTN_MODE_CLTPA == 'before':
                    net = AttnVGG_before(im_size=IM_SIZE, num_classes=2,
                        attention = True, normalize_attn = True, init = 'xavierUniform', interpolate=True)
                elif ATTN_MODE_CLTPA == 'after':
                    net = AttnVGG_after(im_size=IM_SIZE, num_classes=2,
                        attention = True, normalize_attn = True, init = 'xavierUniform', interpolate=True)
                else:
                    raise NotImplementedError("Invalid attention mode!")

                criterion = nn.CrossEntropyLoss()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                device_ids = [0]
                model = nn.DataParallel(net, device_ids=device_ids).to(device)
                criterion.to(device)

                model.load_state_dict(torch.load(CLTPA_MODEL_LIST[count]))
                model.eval()

                with torch.no_grad():
                    pred_numpy = np.zeros((0, 2))
                    labels = np.zeros((0))
                    for k, data in enumerate(testloader, 0):
                        images_test, labels_test = data
                        images_test, labels_test = images_test.to(device), labels_test.to(device)
                        pred_test, __, __, __ = model(images_test)
                        pred_test_numpy = pred_test.cpu().numpy()
                        b = np.zeros_like(pred_test_numpy)
                        b[np.arange(len(pred_test_numpy)), pred_test_numpy.argmax(1)] = 1
                        pred_numpy = np.vstack((pred_numpy, b))
                        labels = np.hstack((labels, labels_test.cpu().numpy()))
                    preds_sum+=pred_numpy
                        # predict = torch.argmax(pred_test, 1)
                        # total += labels_test.size(0)
                        # correct += torch.eq(predict, labels_test).sum().double().item()
                count+=1
    predict = np.argmax(preds_sum, axis=1)
    correct = (predict == labels).sum()
    print("\naccuracy on test data: %.2f%%\n" % (100 * correct / total_test))
    # test data is 78.26