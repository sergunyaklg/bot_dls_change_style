import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import time
from PIL import Image
import copy
import matplotlib.pyplot as plt
from pathlib import Path


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMSIZE = 512


def load_image(image_path):
    loader = transforms.Compose([
        transforms.Resize(IMSIZE),
        transforms.CenterCrop(IMSIZE),
        transforms.ToTensor()])
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)


def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def gram_matrix(input):
    batch_size, f_map_num, h, w = input.size()
    features = input.view(batch_size * f_map_num, h * w)
    G = torch.mm(features, features.t())
    return G.div(batch_size * h * w * f_map_num)


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    
class ImageProcessing:
    def __init__(self, content_image_name, style_image_name):
        self.content_image_name = content_image_name
        self.style_image_name = style_image_name
        self.imsize = 512
        self.loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor()])
        self.device = DEVICE
        self.unloader = transforms.ToPILImage()

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def imshow(self, tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    def load_images(self):
        style_img = self.image_loader(self.style_image_name)
        content_img = self.image_loader(self.content_image_name)
        return style_img, content_img
    
def get_input_optimizer(input_img, optimizer_type):
    if optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS([input_img.requires_grad_()])
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam([input_img.requires_grad_()], lr=0.1)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer_type}.")
    return optimizer

class NST:
    def __init__(self, normalization_mean=None, normalization_std=None, content_layers=None, style_layers=None,
                 style_weight=10000000, content_weight=1, num_steps=300, optimizer_type='LBFGS', cnn=None):
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE) if normalization_mean is None else normalization_mean
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE) if normalization_std is None else normalization_std
        self.content_layers = ['conv_4'] if content_layers is None else content_layers
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'] if style_layers is None else style_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.num_steps = num_steps
        self.optimizer_type = optimizer_type
        self.cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval() if cnn is None else cnn
        self.input_img = None

    def get_style_model_and_losses(self, style_img_tensor, content_img_tensor):
        cnn = copy.deepcopy(self.cnn)
        normalization = Normalization(self.normalization_mean, self.normalization_std).to(DEVICE)
        content_losses = []
        style_losses = []
        model = nn.Sequential(normalization)
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            model.add_module(name, layer)
            if name in self.content_layers:
                target = model(content_img_tensor).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
            if name in self.style_layers:
                target_feature = model(style_img_tensor).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
            if len(content_losses) == len(self.content_layers) and len(style_losses) == len(self.style_layers):
                break
        return model, style_losses, content_losses

    def run_style_transfer(self, content_img_tensor, style_img_tensor, input_img_tensor):
        model, style_losses, content_losses = self.get_style_model_and_losses(style_img_tensor, content_img_tensor)
        optimizer = get_input_optimizer(input_img_tensor, self.optimizer_type)
        run = [0]
        while run[0] <= self.num_steps:
            def closure():
                input_img_tensor.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img_tensor)
                style_score = 0
                content_score = 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                style_score *= self.style_weight
                content_score *= self.content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss: {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
                    print()
                return style_score + content_score
            optimizer.step(closure)
        input_img_tensor.data.clamp_(0, 1)
        return input_img_tensor

    def run(self, style_image_name, content_image_name):
        image_proc = ImageProcessing(content_image_name, style_image_name)
        content_img = image_proc.image_loader(content_image_name)
        input_img = content_img.clone()
        style_img = image_proc.image_loader(style_image_name)
        output = self.run_style_transfer(content_img, style_img, input_img)
        self.input_img = output

    def save_result_as_png(self):
        if self.input_img is None:
            raise TypeError('Nothing to save')
        result_dir = Path('images')
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / 'bot-result.png'
        save_image(self.input_img[0], result_path)
        print(f'Result saved as {result_path}')


def main(content_image_name, style_image_name):
    start_time = time.time()
    print('Creating NST model...')
    nst_model = NST(style_weight=1e7, content_weight=1, optimizer_type='Adam', num_steps=500)
    print('Launching...')
    nst_model.run(style_image_name, content_image_name)
    print("Execution time: %s seconds" % round((time.time() - start_time), 2))
    nst_model.save_result_as_png()
    
    return 0
