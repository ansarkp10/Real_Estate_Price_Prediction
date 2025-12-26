from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class Property(models.Model):
    AREA_TYPES = [
        ('Super built-up Area', 'Super built-up Area'),
        ('Built-up Area', 'Built-up Area'),
        ('Plot Area', 'Plot Area'),
        ('Carpet Area', 'Carpet Area'),
    ]
    
    AVAILABILITY_OPTIONS = [
        ('Ready To Move', 'Ready To Move'),
        ('Immediate Possession', 'Immediate Possession'),
        ('In 1 year', 'In 1 year'),
        ('In 2 years', 'In 2 years'),
        ('In 3 years', 'In 3 years'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    area_type = models.CharField(max_length=50, choices=AREA_TYPES)
    availability = models.CharField(max_length=50, choices=AVAILABILITY_OPTIONS)
    location = models.CharField(max_length=100)
    size = models.CharField(max_length=20)
    society = models.CharField(max_length=100, null=True, blank=True)
    total_sqft = models.CharField(max_length=50)
    bath = models.IntegerField()
    balcony = models.IntegerField()
    predicted_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.size} in {self.location}"
    
    class Meta:
        ordering = ['-created_at']