from django.shortcuts import render, redirect
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
import json

from .models import Property
from .ml_model.model_trainer import predictor

# Custom Forms for better styling
class CustomLoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Username'
        })
        self.fields['password'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Password'
        })

class CustomRegisterForm(UserCreationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Choose a username'
        })
        self.fields['password1'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Create a strong password'
        })
        self.fields['password2'].widget.attrs.update({
            'class': 'form-control',
            'placeholder': 'Confirm your password'
        })

# Authentication Views
def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = CustomRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful! Welcome!")
            return redirect('home')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = CustomRegisterForm()
    
    return render(request, 'estate_app/register.html', {'form': form})

def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    
    if request.method == 'POST':
        form = CustomLoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = CustomLoginForm()
    
    return render(request, 'estate_app/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect('home')

# Main Views
def home(request):
    """Home page - no login required"""
    context = {
        'is_trained': predictor.is_trained,
        'user': request.user,
        'recent_predictions': Property.objects.all().order_by('-created_at')[:3]
    }
    return render(request, 'estate_app/home.html', context)

@login_required
def predict_price(request):
    """Predict price (login required)"""
    prediction = None
    prediction_rupees = None
    form_data = {}
    
    if request.method == 'POST':
        form_data = {
            'area_type': request.POST.get('area_type'),
            'availability': request.POST.get('availability'),
            'location': request.POST.get('location'),
            'size': request.POST.get('size'),
            'society': request.POST.get('society', ''),
            'total_sqft': int(request.POST.get('total_sqft', 0)),
            'bath': int(request.POST.get('bath', 1)),
            'balcony': int(request.POST.get('balcony', 0)),
        }
        
        if predictor.is_trained:
            prediction = predictor.predict(form_data)
            
            if prediction:
                # Convert to rupees for storage
                prediction_rupees = prediction * 100000
                
                # Save prediction
                Property.objects.create(
                    user=request.user,
                    area_type=form_data['area_type'],
                    availability=form_data['availability'],
                    location=form_data['location'],
                    size=form_data['size'],
                    society=form_data['society'],
                    total_sqft=str(form_data['total_sqft']),
                    bath=form_data['bath'],
                    balcony=form_data['balcony'],
                    predicted_price=prediction_rupees
                )
                messages.success(request, f"Prediction saved: ₹{prediction} Lakhs")
        else:
            messages.error(request, "Model not ready. Please try again.")
    
    context = {
        'prediction': prediction,
        'prediction_rupees': prediction_rupees,
        'form_data': form_data,
        'is_trained': predictor.is_trained,
        'area_types': [
            ('Super built-up Area', 'Super built-up Area'),
            ('Built-up Area', 'Built-up Area'),
            ('Plot Area', 'Plot Area'),
            ('Carpet Area', 'Carpet Area'),
        ],
        'availability_options': [
            ('Ready To Move', 'Ready To Move'),
            ('Immediate Possession', 'Immediate Possession'),
            ('In 1 year', 'In 1 year'),
            ('In 2 years', 'In 2 years'),
            ('In 3 years', 'In 3 years'),
        ],
        'locations': ['Bangalore', 'Mumbai', 'Delhi', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata'],
        'sizes': ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK'],
    }
    
    return render(request, 'estate_app/predict.html', context)

@login_required
def my_predictions(request):
    """View user's predictions (login required)"""
    predictions = Property.objects.filter(user=request.user)
    
    context = {
        'predictions': predictions,
        'total_count': predictions.count(),
    }
    
    return render(request, 'estate_app/my_predictions.html', context)

@login_required
def delete_prediction(request, id):
    """Delete a prediction"""
    try:
        prediction = Property.objects.get(id=id, user=request.user)
        prediction.delete()
        messages.success(request, "Prediction deleted successfully!")
    except Property.DoesNotExist:
        messages.error(request, "Prediction not found!")
    
    return redirect('my_predictions')

def api_predict(request):
    """Public API for predictions (no login required)"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            if not predictor.is_trained:
                return JsonResponse({'error': 'Model not ready'}, status=503)
            
            features = {
                'area_type': data.get('area_type', 'Super built-up Area'),
                'availability': data.get('availability', 'Ready To Move'),
                'location': data.get('location', 'Bangalore'),
                'size': data.get('size', '2 BHK'),
                'society': data.get('society', ''),
                'total_sqft': int(data.get('total_sqft', 1000)),
                'bath': int(data.get('bath', 2)),
                'balcony': int(data.get('balcony', 1)),
            }
            
            prediction = predictor.predict(features)
            
            if prediction:
                return JsonResponse({
                    'success': True,
                    'predicted_price_lakhs': prediction,
                    'predicted_price_rupees': prediction * 100000,
                    'message': f'₹{prediction} Lakhs'
                })
            else:
                return JsonResponse({'error': 'Prediction failed'}, status=500)
                
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'error': 'Invalid request method'}, status=405)