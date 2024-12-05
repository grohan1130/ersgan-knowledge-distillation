import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train_student_model(student_model, dataloader, teacher_model, num_epochs=10, device="cuda"):
    """Train the student model with teacher supervision."""
    student_model = student_model.to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    pixel_loss = torch.nn.L1Loss()

    for epoch in range(num_epochs):
        for lr_img, hr_img in dataloader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            with torch.no_grad():
                teacher_output = teacher_model.predict(lr_img.cpu().numpy())
            teacher_output = torch.tensor(teacher_output).to(device)
            student_output = student_model(lr_img)

            loss = pixel_loss(student_output, teacher_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    return student_model
