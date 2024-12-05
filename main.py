import torch
from dataset import SuperResolutionDataset
from models import StudentSRNet
from train import train_student_model
from utils import calculate_psnr, calculate_ssim
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize

if __name__ == "__main__":
    # Define paths
    lr_dir = "./data/LR"
    hr_dir = "./data/HR"

    # Dataset and DataLoader
    transform = Compose([
        CenterCrop(128),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = SuperResolutionDataset(lr_dir, hr_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Models
    student_model = StudentSRNet()
    teacher_model = torch.load("path_to_teacher_model.pth")  # Load pre-trained teacher model

    # Train student model
    trained_model = train_student_model(student_model, dataloader, teacher_model, num_epochs=10)

    # Save the trained model
    torch.save(trained_model.state_dict(), "student_srnet.pth")
