from django.db import models

class PlantImage(models.Model):
    photo = models.ImageField(upload_to='plants/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"PlantImage {self.id}"
