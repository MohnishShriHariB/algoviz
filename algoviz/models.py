from django.db import models


class pics(models.Model):
    frame=models.ImageField(upload_to="midea/algoviz/images/")
    gif=models.ImageField(upload_to="midea/algoviz/images/")
    name=models.CharField(max_length=100)
    fno=models.IntegerField()
    def __str__(self):
        return self.name
