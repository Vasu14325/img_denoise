from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import torch
from .forms import UploadImageForm
from .model import Autoencoder
from PIL import Image
import torchvision.transforms as transforms
import os

device = torch.device('cpu')
model_path = os.path.join(settings.BASE_DIR, 'autoencoder_model498.pth')

loaded_autoencoder = Autoencoder().to(device)
loaded_autoencoder.load_state_dict(torch.load(model_path, map_location=device))
loaded_autoencoder.eval()

def index(request):
    form = UploadImageForm()
    return render(request, 'denoise/index.html', {'form': form})

def upload(request):
    if request.method == 'POST' and request.FILES['image']:
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_file_url = fs.url(filename)

            # Load and resize the image for display
            img = Image.open(os.path.join(settings.MEDIA_ROOT, filename)).convert('RGB')
            img_resized = img.resize((400, 400))
            resized_img_path = os.path.join(settings.MEDIA_ROOT, 'resized_' + filename)
            img_resized.save(resized_img_path)

            # Resize the image for processing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                denoised_img_tensor = loaded_autoencoder(img_tensor).cpu()
            
            denoised_img = transforms.ToPILImage()(denoised_img_tensor.squeeze(0))

            # Resize the denoised image for display
            denoised_img = denoised_img.resize((400, 400))
            denoised_img_path = os.path.join(settings.MEDIA_ROOT, 'denoised_' + filename)
            denoised_img.save(denoised_img_path)

            return render(request, 'denoise/result.html', {
                'form': form,
                'uploaded_file_url': fs.url('resized_' + filename),
                'denoised_img_url': fs.url('denoised_' + filename)
            })
    else:
        form = UploadImageForm()
    return render(request, 'denoise/index.html', {'form': form})
