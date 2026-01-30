from u_net import UNet
from pycocotools.coco import COCO
import numpy as np
import torch
from utils import COCOSegDataset, get_transform, mean_iou
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

TOP_K_CATEGORIES = 3
TOP_K_IMAGES_TRAINING = 10000
TOP_K_IMAGES_VALID = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(data_path, annotations_path, epochs, batch_size, learning_rate):
    coco = COCO(annotations_path)

    # keep only top K categories
    cat_ids = coco.getCatIds()
    cat_counts = {cat_id: len(coco.getAnnIds(catIds=cat_id)) for cat_id in cat_ids}
    top_k_cat_ids = sorted(cat_counts, key=cat_counts.get, reverse=True)[:TOP_K_CATEGORIES]

    img_ids = coco.getImgIds(catIds=top_k_cat_ids)[:TOP_K_IMAGES_TRAINING]
    train_dataset = COCOSegDataset(data_path, coco, img_ids, top_k_cat_ids, transform=get_transform())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)                                                 

    valid_img_ids = coco.getImgIds(catIds=top_k_cat_ids)[TOP_K_IMAGES_TRAINING:TOP_K_IMAGES_TRAINING+TOP_K_IMAGES_VALID]
    valid_dataset = COCOSegDataset(data_path, coco, valid_img_ids, top_k_cat_ids, transform=get_transform())
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    nb_labels = TOP_K_CATEGORIES + 1  # including background
    
    model = UNet(in_channels=3, nb_labels=nb_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_v_loss = np.inf
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_iou = 0.0
        for images, masks in tqdm(train_loader):        
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iou += mean_iou(outputs, masks, nb_labels)

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        avg_train_iou = total_iou / len(train_loader)
        writer.add_scalar("IoU/train", avg_train_iou, epoch)

        # Validation
        model.eval()
        total_v_loss = 0.0
        total_v_iou = 0.0
        with torch.no_grad():
            for v_images, v_masks in valid_loader:
                v_images = v_images.to(device)
                v_masks = v_masks.to(device)

                v_outputs = model(v_images)
                v_loss = criterion(v_outputs, v_masks)
                total_v_loss += v_loss.item()
                total_v_iou += mean_iou(v_outputs, v_masks, nb_labels)

        avg_v_loss = total_v_loss / len(valid_loader)
        writer.add_scalar("Loss/valid", avg_v_loss, epoch)
        avg_v_iou = total_v_iou / len(valid_loader)
        writer.add_scalar("IoU/valid", avg_v_iou, epoch)

        if avg_v_loss < best_v_loss:
            best_v_loss = avg_v_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("Saved Best Model")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_v_loss:.4f}")
        torch.save(model.state_dict(), "last_unet_model.pth")
        print("Saved Last Model")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train U-Net model on coco 2014 dataset")
    
    parser.add_argument('--data_path', '-p', type=str, required=True, help='Path to COCO 2014 dataset')
    parser.add_argument('--annotations_path', '-a', type=str, required=True, help='Path to COCO 2014 annotations')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='Learning rate for optimizer')

    args = parser.parse_args()
    main(
        data_path=args.data_path,
        annotations_path=args.annotations_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )