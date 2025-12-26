from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from .models import Property, UserProfile

class CustomUserCreationForm(UserCreationForm):
    """Custom user registration form"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'})
    )
    first_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First Name'})
    )
    last_name = forms.CharField(
        max_length=30,
        required=True,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last Name'})
    )
    
    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Username'})
        self.fields['password1'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Password'})
        self.fields['password2'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Confirm Password'})
    
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("This email is already registered.")
        return email

class CustomAuthenticationForm(AuthenticationForm):
    """Custom login form"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Username'})
        self.fields['password'].widget.attrs.update({'class': 'form-control', 'placeholder': 'Password'})

class UserProfileForm(forms.ModelForm):
    """User profile form"""
    class Meta:
        model = UserProfile
        fields = ['phone', 'address', 'city', 'profile_picture']
        widgets = {
            'phone': forms.TextInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'city': forms.TextInput(attrs={'class': 'form-control'}),
            'profile_picture': forms.FileInput(attrs={'class': 'form-control'}),
        }

class UserUpdateForm(forms.ModelForm):
    """User update form"""
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']
        widgets = {
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
        }

class PropertyForm(forms.ModelForm):
    """Property form"""
    class Meta:
        model = Property
        fields = [
            'area_type', 'availability', 'location', 'size',
            'society', 'total_sqft', 'bath', 'balcony', 'price'
        ]
        widgets = {
            'area_type': forms.Select(attrs={'class': 'form-control'}),
            'availability': forms.Select(attrs={'class': 'form-control'}),
            'location': forms.TextInput(attrs={'class': 'form-control'}),
            'size': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 2 BHK'}),
            'society': forms.TextInput(attrs={'class': 'form-control'}),
            'total_sqft': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 1200 or 1000-1500'}),
            'bath': forms.NumberInput(attrs={'class': 'form-control'}),
            'balcony': forms.NumberInput(attrs={'class': 'form-control'}),
            'price': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'placeholder': 'Price in lakhs'}),
        }
        labels = {
            'price': 'Actual Price (in lakhs)'
        }

class PredictionForm(forms.Form):
    """Prediction form"""
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
    
    area_type = forms.ChoiceField(
        choices=AREA_TYPES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    availability = forms.ChoiceField(
        choices=AVAILABILITY_OPTIONS,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    location = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Whitefield, Indiranagar'
        })
    )
    size = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 2 BHK, 3 BHK'
        })
    )
    society = forms.CharField(
        max_length=100,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., Prestige, Sobha (optional)'
        })
    )
    total_sqft = forms.CharField(
        max_length=50,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 1200 or 1000-1500'
        })
    )
    bath = forms.IntegerField(
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    balcony = forms.IntegerField(
        min_value=0,
        max_value=5,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    def clean_total_sqft(self):
        data = self.cleaned_data['total_sqft']
        import re
        if not re.match(r'^\d+(\.\d+)?(-\d+(\.\d+)?)?$', data):
            raise forms.ValidationError("Enter valid sqft (e.g., 1200 or 1000-1500)")
        return data
    
    def clean_size(self):
        data = self.cleaned_data['size']
        import re
        if not re.match(r'^\d+\s*BHK$', data, re.IGNORECASE):
            raise forms.ValidationError("Enter valid size (e.g., 2 BHK)")
        return data