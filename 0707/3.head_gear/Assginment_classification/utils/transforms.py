# ./utils/transforms.py

from torchvision import transforms

transform = transforms.Compose(
                transforms.Resize(224),
                transforms.TenCrop(32),
                transforms.Normalize((0.5,),(0.5,)),
                transforms.ToTensor(),
                               )

target_transform = transforms.Compose(
                    transforms.Normalize((0.5,),(0.5,)),
                    transforms.ToTensor(),
                    )


# transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(15),
#         transforms.CenterCrop(64),
#         transforms.GaussianBlur(5),
#         transforms.ToTensor(),
#     ])                    