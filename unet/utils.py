import os
import cv2
import torch
import numpy as np
import albumentations as A


def get_transform():
    return A.Compose([
        A.Resize(128, 128),
    ])

def mean_iou(predictions, targets, nb_classes):
    preds = torch.argmax(predictions, dim=1)
    ious = []
    for cls in range(1, nb_classes):
        pred_mask = preds == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 0.0


class COCOSegDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, coco, img_ids, cat_ids=None, transform=None):
        self.img_dir = img_dir
        self.coco = coco
        self.img_ids = img_ids
        self.cat_ids = cat_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = self.coco_to_mask(img_id)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).long()

        return image, mask

    def coco_to_mask(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        h, w = img_info["height"], img_info["width"]
        
        mask = np.zeros((h, w), dtype=np.uint8)

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            category_id = ann["category_id"]
            m = self.coco.annToMask(ann)
            if self.cat_ids and category_id not in self.cat_ids:
                continue
            cat_index = self.cat_ids.index(category_id) + 1 if self.cat_ids else category_id
            mask[m == 1] = cat_index

        return mask