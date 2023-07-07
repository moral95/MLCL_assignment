{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 'test.ipynb'\n",
    "import torch\n",
    "import torchvision.utils as vutils\n",
    "from utils.DCGAN import Generator\n",
    "from torchvision import transforms\n",
    "from torch.utils.data import DataLoader\n",
    "from utils.dataset import HeadGearDataset\n",
    "from utils.config import load_config\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "target_transform = transforms.Compose(\n",
    "                    transforms.Normalize((0.5,),(0.5,)),\n",
    "                    transforms.ToTensor()\n",
    "                    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def test(config):\n",
    "    device = torch.device(\"cuda:0\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "    transform = transforms.Compose([\n",
    "        transforms.Resize(config['model']['image_size']),\n",
    "        transforms.CenterCrop(config['model']['image_size']),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),\n",
    "    ])\n",
    "\n",
    "    # TODO: Create the test dataset\n",
    "    test_data = HeadGearDataset(config['paths']['annotation'], config['paths']['dataset_path'], mode='train', transform=transform)\n",
    "    test_loader = DataLoader(test_data, batch_size= config['training']['batch_size'], shuffle=True)\n",
    "\n",
    "    \n",
    "    # Loading the trained generator\n",
    "    netG = Generator(config['training']['ngpu'],\n",
    "                     config['model']['nz'],\n",
    "                     config['model']['nc'],\n",
    "                     config['model']['ngf']\n",
    "                     ).to(device)\n",
    "    \n",
    "    # TODO: Load the trained model\n",
    "    # fiil this in\n",
    "    netG.load_state_dict(torch.load(config['paths']['model_save_path']))\n",
    "\n",
    "    # TODO: Set the model to evaluation mode\n",
    "    netG.eval()\n",
    "\n",
    "    real_batch = next(iter(test_loader))\n",
    "    \n",
    "    # Plotting real images\n",
    "    plt.figure(figsize=(15,15))\n",
    "    plt.subplot(1,2,1)\n",
    "    plt.axis(\"off\")\n",
    "    plt.title(\"Real Images\")\n",
    "    plt.imshow(transforms.ToPILImage()(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu()))\n",
    "\n",
    "    # Generating fake images and plotting them\n",
    "    noise = torch.randn(8, config['model']['nz'], 1, 1, device=device)\n",
    "    fake_images = netG(noise).detach().cpu()\n",
    "    plt.subplot(1,2,2)\n",
    "    plt.axis(\"off\")\n",
    "    plt.title(\"Fake Images\")\n",
    "    plt.imshow(transforms.ToPILImage()(vutils.make_grid(fake_images, padding=5, normalize=True)))\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "config = load_config('configs/configs.yaml')  # specify the path to your config file\n",
    "test(config)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "mlcl",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.16"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
