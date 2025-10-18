from django.urls import path
from . import views

app_name = 'app_uploads'

urlpatterns = [
    path('', views.upload_view, name='upload'),
    # path('result/', views.result_view, name='result'),  <-- REMOVE THIS LINE
]
