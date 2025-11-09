from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import os
from .data_analyzer import AnalizadorDataset

DATASETS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
DATASET_NAME = "diabetes.csv"

analizador_global = AnalizadorDataset(DATASETS_PATH, DATASET_NAME)

@api_view(['GET'])
def cargar_dataset(request):
    try:
        resultado = analizador_global.cargar_dataset()
        
        if resultado['success']:
            return Response({
                'success': True,
                'mensaje': 'Dataset cargado exitosamente',
                'datos': resultado
            })
        else:
            return Response({
                'success': False,
                'error': resultado['error']
            }, status=400)
            
    except Exception as e:
        return Response({
            'success': False,
            'error': f'Error al cargar dataset: {str(e)}'
        }, status=500)

@api_view(['GET'])
def visualizar_dataset(request):
    try:
        resultado = analizador_global.visualizar_dataset()
        
        if resultado['success']:
            return Response(resultado)
        else:
            return Response({
                'success': False,
                'error': resultado['error']
            }, status=400)
            
    except Exception as e:
        return Response({
            'success': False,
            'error': f'Error en visualización: {str(e)}'
        }, status=500)

@api_view(['GET'])
def importancia_caracteristicas(request):
    try:
        resultado = analizador_global.importancia_caracteristicas()
        
        if resultado['success']:
            return Response(resultado)
        else:
            return Response({
                'success': False,
                'error': resultado['error']
            }, status=400)
            
    except Exception as e:
        return Response({
            'success': False,
            'error': f'Error al calcular importancia: {str(e)}'
        }, status=500)

@api_view(['GET'])
def reducir_caracteristicas(request):
    try:
        resultado = analizador_global.reducir_caracteristicas()
        
        if resultado['success']:
            return Response(resultado)
        else:
            return Response({
                'success': False,
                'error': resultado['error']
            }, status=400)
            
    except Exception as e:
        return Response({
            'success': False,
            'error': f'Error al reducir características: {str(e)}'
        }, status=500)

@api_view(['GET'])
def calcular_f1_score(request):
    try:
        resultado = analizador_global.calcular_f1_score()
        
        if resultado['success']:
            return Response(resultado)
        else:
            return Response({
                'success': False,
                'error': resultado['error']
            }, status=400)
            
    except Exception as e:
        return Response({
            'success': False,
            'error': f'Error al calcular F1 score: {str(e)}'
        }, status=500)

@api_view(['POST'])
def limpiar_cache(request):
    try:
        success = analizador_global.limpiar_cache()
        if success:
            return Response({
                'success': True,
                'mensaje': 'Cache y modelos limpiados exitosamente'
            })
        else:
            return Response({
                'success': False,
                'error': 'Error al limpiar cache'
            }, status=400)
    except Exception as e:
        return Response({
            'success': False,
            'error': f'Error al limpiar cache: {str(e)}'
        }, status=500)

def frontend_view(request):
    return render(request, 'index.html')
