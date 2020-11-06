import torchvision.transforms as transforms

from model_ltpa import AttnVGG_before, AttnVGG_after
from utilities import *
from dataset import BlightDataset
from config import *


from PIL import Image

if __name__ == '__main__':

    transform_test = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE), interpolation=Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.36534598,0.36992216,0.33446208), (0.2657754 , 0.26731136, 0.2778652))
    ])

    testset = BlightDataset(root='data', set_type='test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=5)

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

    criterion = nn.CrossEntropyLoss()
    print('done')

    print('\nmoving to GPU ...\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    print('done')

    model.load_state_dict(torch.load(MODEL_DIR_LTPA))


    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        # log scalars
        images_disp = []
        for i, data in enumerate(testloader, 0):
            images_test, labels_test = data
            images_test, labels_test = images_test.to(device), labels_test.to(device)
            #images_disp.append(images_test[0:1, :, :, :])
            pred_test, __, __, __ = model(images_test)
            predict = torch.argmax(pred_test, 1)
            total += labels_test.size(0)
            correct += torch.eq(predict, labels_test).sum().double().item()
        print("\naccuracy on test data: %.2f%%\n" % (100 * correct / total))
        print('\nlog images ...\n')
        # I_train = utils.make_grid(images_disp[0], nrow=6, normalize=True, scale_each=True)
        # I_test = utils.make_grid(images_disp[1], nrow=6, normalize=True, scale_each=True)
        '''print('\nlog attention maps ...\n')
        min_up_factor = 1
        vis_fun = visualize_attn_softmax
        # training data
        __, c1, c2, c3 = model(images_disp[0])
        if c1 is not None:
            attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=6)

        if c2 is not None:
            attn2 = vis_fun(I_train, c2, up_factor=min_up_factor * 2, nrow=6)
        if c3 is not None:
            attn3 = vis_fun(I_train, c3, up_factor=min_up_factor * 4, nrow=6)

        # test data
        __, c1, c2, c3 = model(images_disp[1])
        if c1 is not None:
            attn1 = vis_fun(I_test, c1, up_factor=min_up_factor, nrow=6)

        if c2 is not None:
            attn2 = vis_fun(I_test, c2, up_factor=min_up_factor * 2, nrow=6)

        if c3 is not None:
            attn3 = vis_fun(I_test, c3, up_factor=min_up_factor * 4, nrow=6)'''
