from django.shortcuts import render
from django.views.generic.edit import CreateView
from .models import ImageModel
from .forms import ImageUploadForm
from PIL import Image as im
import io, torch

# Create your views here.

def homepage(self, request, *args, **kwargs):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        img = request.FILES.get('image')
        img_instance = ImageModel(
            image=img
        )
        img_instance.save()

        uploaded_img_qs = ImageModel.objects.filter().last()
        img_bytes = uploaded_img_qs.image.read()
        img = im.open(io.BytesIO(img_bytes))

        # Change this to the correct path
        path_hubconfig = "yolov5_SR71/hubconf.py"
        path_weightfile = "yolov5_SR71/runs/train/exp4/weights/best.pt"  # or any custom trained model

        model = torch.hub.load(path_hubconfig, 'custom',
                            path=path_weightfile, source='local')

        results = model(img, size=640)
        results.render()
        for img in results.imgs:
            img_base64 = im.fromarray(img)
            img_base64.save("media/yolo_out/image0.jpg", format="JPEG")

        inference_img = "/media/yolo_out/image0.jpg"

        form = ImageUploadForm()
        context = {
            "form": form,
            "inference_img": inference_img
        }
        return render(request, 'image/imagemodel_form.html', context)

    else:
        form = ImageUploadForm()
    context = {
        "form": form
    }
    return render(request, 'image/imagemodel_form.html', context)
