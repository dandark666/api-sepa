from django.contrib import admin
from django.urls import path, include
from analizador.views import frontend_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', frontend_view, name='home'),
    path('', include('analizador.urls')),
]