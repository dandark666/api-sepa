from django.urls import path
from . import views

urlpatterns = [
    path('api/cargar-dataset/', views.cargar_dataset, name='cargar_dataset'),
    path('api/visualizar-dataset/', views.visualizar_dataset, name='visualizar_dataset'),
    path('api/importancia-caracteristicas/', views.importancia_caracteristicas, name='importancia_caracteristicas'),
    path('api/reducir-caracteristicas/', views.reducir_caracteristicas, name='reducir_caracteristicas'),
    path('api/calcular-f1-score/', views.calcular_f1_score, name='calcular_f1_score'),
    path('api/limpiar-cache/', views.limpiar_cache, name='limpiar_cache'),
]