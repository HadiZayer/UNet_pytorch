from torchvision import transforms
import random

"""
Input tensors w/ size (3xHxW)
Output tensor w/ size (Ax3xHxW) where A is the # of augmentations
A = rotationNum * cropNum

augments the input with random rotations, croping, and flipping
"""

def augment_data(input_img, label, rotationNum=5, cropNum=5, hflip=True, vflip=True):
    
    input_pil = transforms.ToPILImage()(input_img)
    label_pil = transforms.ToPILImage()(label)
    
    
    modified_inputs = []
    modified_labels = []
    
    for r in range(rotationNum):
        degrees = random.uniform(-90,90)
        rotTransform = transforms.RandomRotation(degrees=(degrees, degrees))
        modified_input = rotTransform(input_pil)
        modified_label = rotTransform(label_pil)
        for c in range(cropNum):
            scale = random.uniform(0.7,1)
            ratio = random.uniform(3/4,4/3)
            cropTransform = transforms.RandomResizedCrop(size=input_pil.size[0], scale=(scale,scale), ratio=(ratio, ratio))
            modified_input = cropTransform(modified_input)
            modified_label = cropTransform(modified_label)
            
            if hflip:
                if(random.random() >= 0.5):
                    hflipTransform = transforms.RandomHorizontalFlip(p=1)
                    modified_input = hflipTransform(modified_input)
                    modified_label = hflipTransform(modified_label)
                
            if vflip:
                if(random.random() >= 0.5):
                    vflipTransform = transforms.RandomVerticalFlip(p=1)
                    modified_input = vflipTransform(modified_input)
                    modified_label = vflipTransform(modified_label)
                
            modified_inputs.append(transforms.ToTensor()(modified_input))
            modified_labels.append(transforms.ToTensor()(modified_label))
    return modified_inputs, modified_labels