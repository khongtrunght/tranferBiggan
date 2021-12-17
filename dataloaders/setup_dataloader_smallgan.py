import glob
from .ImageListDataset import ImageListDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def setup_dataloader(name, h=128, w=128, batch_size=4, num_workers=4):
    '''
    instead of setting up dataloader that read raw image from file, 
    let's use store all images on cpu memmory
    because this is for small dataset
    '''
    name = 'animal'
    data_size = 45

    if name == "face":
        img_path_list = glob.glob("./data/face/*.png")
    elif name == "anime":
        img_path_list = glob.glob("./data/anime/*.png")
    else:
        img_path_dict = {}

        labels_dict = {'cat': 0, 'dog': 1, 'wild': 2}
        for label in labels_dict.keys():
            img_path_dict[label] = glob.glob(
                f"./data/afhq/train/{label}/*.jpg")
            img_path_dict[label] = img_path_dict[label][:data_size//3]

        img_path_list = []

        for label in labels_dict.keys():
            img_path_list.extend([(path, labels_dict[label])
                                  for path in img_path_dict[label]])

        # tra lai label i la so thu tu, data[1] la label

    assert len(img_path_list) > 0

    transform = transforms.Compose([
        transforms.Resize(min(h, w)),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
    ])

    img_path_list = [(data[0], (i, data[1]))
                     for i, data in enumerate(sorted(img_path_list))]

    dataset = ImageListDataset(
        img_path_list, transform=transform)

    return DataLoader([data for data in dataset], batch_size=batch_size,
                      shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
