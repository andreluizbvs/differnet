from model import load_model, load_weights
from os import listdir
from os.path import join
from utils import get_random_transforms, get_fixed_transforms
from PIL import Image
import config as c
import model as m
import torch

def get_anomaly_score(model, image_path, transforms):
    img = Image.open(image_path).convert('RGB')
    transformed_imgs = torch.stack([tf(img) for tf in transforms])
    transformed_imgs = transformed_imgs.to(c.device)
    z = model(transformed_imgs)
    anomaly_score = torch.mean(z ** 2)
    print("image: %s, score: %.2f" % (image_path, anomaly_score))
    return anomaly_score

def infer(model_name, image_folder, fixed_transforms=True):
    model = m.DifferNet()
    model = load_weights(model, model_name)
    model.to(c.device)
    files = listdir(image_folder)

    if fixed_transforms:
        fixed_degrees = [i * 360.0 / c.n_transforms_test for i in range(c.n_transforms_test)]
        transforms = [get_fixed_transforms(fd) for fd in fixed_degrees]
    else:
        transforms = [get_random_transforms()] * c.n_transforms_test

    for f in files:
        get_anomaly_score(model, join(image_folder, f), transforms)

infer("/best.weights", c.image_folder, fixed_transforms=True)