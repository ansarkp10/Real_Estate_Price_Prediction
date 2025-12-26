from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Main Pages
    path('', views.home, name='home'),
    path('predict/', views.predict_price, name='predict'),
    path('my-predictions/', views.my_predictions, name='my_predictions'),
    path('delete-prediction/<int:id>/', views.delete_prediction, name='delete_prediction'),
    
    # API
    path('api/predict/', views.api_predict, name='api_predict'),
]