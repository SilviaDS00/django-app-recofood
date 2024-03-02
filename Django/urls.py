"""
URL configuration for Django project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from app.views import chatbot_view, prediction, prediction_bmi, options

urlpatterns = [
    path("", chatbot_view, name="chatbot_view"),  # Ruta para la página de inicio
    path("chatbot/", chatbot_view, name="chatbot"),  # Ruta para el chatbot
    path("prediction/", prediction, name="prediction"), # Ruta para la predicción de la imagen
    path("prediction-bmi/", prediction_bmi, name="prediction_bmi"), # Ruta para la predicción del BMI
    path("options/", options, name="options") # Ruta para las opciones
]