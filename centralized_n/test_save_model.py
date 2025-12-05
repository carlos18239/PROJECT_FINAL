"""
Script de prueba para verificar el guardado del mejor modelo
"""
import torch
import os
from mlp import MLP

def test_load_saved_model():
    """Prueba cargar un modelo guardado"""
    
    # Buscar modelos guardados
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("No hay directorio de modelos aún")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    if not model_files:
        print("No hay modelos guardados aún")
        return
    
    print(f"Modelos encontrados: {len(model_files)}")
    for f in model_files:
        print(f"  - {f}")
    
    # Cargar el primer modelo como ejemplo
    model_path = os.path.join(model_dir, model_files[0])
    print(f"\nCargando modelo: {model_path}")
    
    checkpoint = torch.load(model_path)
    
    print("\nInformación del modelo:")
    print(f"  Round: {checkpoint.get('round', 'N/A')}")
    print(f"  Recall: {checkpoint.get('recall', 'N/A'):.4f}")
    print(f"  Agent ID: {checkpoint.get('agent_id', checkpoint.get('aggregator_id', 'N/A'))}")
    
    if 'recall_history' in checkpoint:
        history = checkpoint['recall_history']
        print(f"\n  Historial de recall ({len(history)} rondas):")
        for h in history[-5:]:  # Últimas 5 rondas
            print(f"    Ronda {h['round']}: {h['recall']:.4f}")
    
    # Verificar que los parámetros del modelo se pueden cargar
    model_state = checkpoint['model_state_dict']
    print(f"\n  Parámetros del modelo: {len(model_state)} tensores")
    for name, tensor in list(model_state.items())[:3]:
        print(f"    {name}: {tensor.shape}")
    
    print("\n✓ Modelo cargado exitosamente")

if __name__ == '__main__':
    test_load_saved_model()
