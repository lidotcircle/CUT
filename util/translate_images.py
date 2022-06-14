import torch
import torch.nn as nn
import os
from torchvision import transforms
from data.image_folder import ImageFolder
from torchvision.utils import save_image


def translate_images(generator: nn.Module, src_images_dir: str, dst_images_dir: str, device = 'cpu', batch_size = 50):
    images = ImageFolder(src_images_dir, return_paths=True, transform=transforms.Compose([transforms.ToTensor()]))
    num_images = len(images)

    if not os.path.isdir(dst_images_dir):
        os.makedirs(dst_images_dir)

    with torch.no_grad():
        image_values = []
        image_names = []
        start_pos = 0
        while start_pos < num_images:
            num_get = min(batch_size, num_images - start_pos)
            for i in range(num_get):
                img, img_path = images[start_pos + i]
                image_values.append(img)
                image_names.append(os.path.basename(img_path))
            src = torch.stack(image_values).to(device)
            tgt = generator(src)
            if isinstance(tgt, tuple):
                tgt, _ = tgt

            for i, name in enumerate(image_names):
                img = tgt[i]
                dst = os.path.join(dst_images_dir, name)
                save_image(img, dst)

            image_values = []
            image_names = []
            start_pos += num_get