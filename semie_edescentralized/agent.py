"""
Agente para Aprendizaje Federado Semi-Descentralizado
Puede actuar como worker (entrena y envía) o como agregador (recibe, agrega y distribuye)
"""
import asyncio
import websockets
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import sys
from typing import Dict, List

# Importar configuración
# Para Raspberry Pi, cambiar a: from config import RaspberryPiConfig as Config
from config import RaspberryPiConfig as Config


class SimpleModel(nn.Module):
    """Modelo simple para demostración"""
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FederatedAgent:
    def __init__(self, agent_id, server_uri='ws://localhost:8765'):
        self.agent_id = agent_id
        self.server_uri = server_uri
        self.server_ws = None
        self.role = None  # 'worker' o 'aggregator'
        self.aggregator_info = None
        
        # Modelo y datos
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.train_data = None
        
        # Para modo agregador
        self.aggregator_server = None
        self.received_models = {}
        self.expected_agents = []
        self.worker_connections = {}  # Almacenar conexiones de workers
        
        # Puerto para recibir modelos cuando es agregador
        self.aggregator_port = 9000 + int(agent_id.split('_')[1])
        
    def create_local_dataset(self, num_samples=None):
        """Crea un dataset local sintético para el agente"""
        if num_samples is None:
            num_samples = Config.DATASET_SIZE
        
        # Datos sintéticos diferentes para cada agente
        np.random.seed(int(self.agent_id.split('_')[1]))
        X = np.random.randn(num_samples, Config.INPUT_SIZE).astype(np.float32)
        y = np.random.randint(0, Config.OUTPUT_SIZE, num_samples)
        
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
        self.train_data = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        print(f"[{self.agent_id}] Dataset local creado con {num_samples} muestras")
    
    def train_local_model(self, epochs=5):
        """Entrena el modelo con datos locales"""
        print(f"[{self.agent_id}] Iniciando entrenamiento local ({epochs} épocas)...")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in self.train_data:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_data)
            print(f"[{self.agent_id}] Época {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print(f"[{self.agent_id}] ✓ Entrenamiento local completado")
    
    def get_model_parameters(self):
        """Obtiene los parámetros del modelo"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters):
        """Establece los parámetros del modelo"""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()
    
    async def send_model_to_aggregator(self, aggregator_info):
        """Envía el modelo local al agregador"""
        aggregator_uri = f"ws://{aggregator_info['host']}:{aggregator_info['port']}"
        max_retries = Config.MAX_RETRIES
        retry_delay = Config.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                print(f"[{self.agent_id}] Conectando con agregador en {aggregator_uri}... (intento {attempt + 1}/{max_retries})")
                
                async with websockets.connect(aggregator_uri) as ws:
                    # Serializar parámetros del modelo
                    model_params = self.get_model_parameters()
                    params_bytes = pickle.dumps(model_params)
                    
                    message = {
                        'type': 'model_update',
                        'agent_id': self.agent_id,
                        'params_size': len(params_bytes)
                    }
                    
                    # Enviar metadata
                    await ws.send(json.dumps(message))
                    
                    # Enviar parámetros
                    await ws.send(params_bytes)
                    
                    print(f"[{self.agent_id}] ✓ Modelo enviado al agregador ({len(params_bytes)} bytes)")
                    
                    # Esperar modelo agregado
                    print(f"[{self.agent_id}] Esperando modelo agregado...")
                    response = await ws.recv()
                    
                    if isinstance(response, bytes):
                        aggregated_params = pickle.loads(response)
                        self.set_model_parameters(aggregated_params)
                        print(f"[{self.agent_id}] ✓ Modelo agregado recibido y actualizado")
                    
                    return  # Éxito, salir de la función
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[{self.agent_id}] Error al conectar (intento {attempt + 1}): {e}")
                    print(f"[{self.agent_id}] Reintentando en {retry_delay} segundos...")
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"[{self.agent_id}] Error al comunicarse con agregador después de {max_retries} intentos: {e}")
    
    async def handle_worker_connection(self, websocket, path):
        """Maneja conexiones de workers cuando este agente es agregador"""
        worker_id = None
        try:
            # Recibir metadata
            metadata = await websocket.recv()
            data = json.loads(metadata)
            
            if data['type'] == 'model_update':
                worker_id = data['agent_id']
                print(f"[{self.agent_id}] Recibiendo modelo de {worker_id}...")
                
                # Recibir parámetros
                params_bytes = await websocket.recv()
                model_params = pickle.loads(params_bytes)
                
                self.received_models[worker_id] = model_params
                self.worker_connections[worker_id] = websocket  # Guardar conexión
                print(f"[{self.agent_id}] ✓ Modelo de {worker_id} recibido ({len(self.received_models)}/{len(self.expected_agents)})")
                
                # Si recibimos todos los modelos, agregar y distribuir
                if len(self.received_models) == len(self.expected_agents):
                    print(f"[{self.agent_id}] Todos los modelos recibidos. Iniciando agregación...")
                    aggregated_params = self.federated_averaging()
                    
                    # Actualizar modelo propio
                    self.set_model_parameters(aggregated_params)
                    
                    # Enviar modelo agregado a TODOS los workers
                    await self.distribute_aggregated_model(aggregated_params)
                    
                    # Notificar al servidor
                    await self.notify_server_aggregation_complete()
                else:
                    # Esperar hasta que se complete la agregación
                    for _ in range(30):  # Esperar máximo 30 segundos
                        if hasattr(self, 'final_aggregated_params'):
                            break
                        await asyncio.sleep(1)
                
                # Enviar modelo agregado a este worker
                if hasattr(self, 'final_aggregated_params'):
                    params_bytes = pickle.dumps(self.final_aggregated_params)
                    await websocket.send(params_bytes)
                    print(f"[{self.agent_id}] ✓ Modelo agregado enviado a {worker_id}")
                    
        except Exception as e:
            print(f"[{self.agent_id}] Error manejando worker {worker_id}: {e}")
    
    def federated_averaging(self):
        """Implementa FedAvg para agregar modelos"""
        print(f"[{self.agent_id}] Ejecutando FedAvg con {len(self.received_models)} modelos...")
        
        # Agregar modelo propio a la lista
        all_models = list(self.received_models.values())
        all_models.append(self.get_model_parameters())
        
        # Inicializar parámetros agregados
        aggregated_params = {}
        
        # Promediar cada parámetro
        for param_name in all_models[0].keys():
            # Sumar todos los parámetros
            param_sum = torch.zeros_like(all_models[0][param_name])
            for model_params in all_models:
                param_sum += model_params[param_name]
            
            # Calcular promedio
            aggregated_params[param_name] = param_sum / len(all_models)
        
        print(f"[{self.agent_id}] ✓ FedAvg completado")
        self.final_aggregated_params = aggregated_params
        return aggregated_params
    
    async def distribute_aggregated_model(self, aggregated_params):
        """Distribuye el modelo agregado a todos los workers"""
        print(f"[{self.agent_id}] Modelo agregado listo para distribución")
        # La distribución se hace en handle_worker_connection cuando cada worker recibe su respuesta
    
    async def notify_server_aggregation_complete(self):
        """Notifica al servidor que la agregación está completa"""
        if self.server_ws:
            message = {
                'type': 'aggregation_complete',
                'agent_id': self.agent_id
            }
            await self.server_ws.send(json.dumps(message))
    
    async def start_aggregator_server(self):
        """Inicia servidor para recibir modelos de workers"""
        # Cerrar servidor anterior si existe
        if self.aggregator_server:
            self.aggregator_server.close()
            await self.aggregator_server.wait_closed()
            print(f"[{self.agent_id}] Servidor agregador anterior cerrado")
        
        print(f"[{self.agent_id}] Iniciando servidor agregador en puerto {self.aggregator_port}...")
        
        self.aggregator_server = await websockets.serve(
            self.handle_worker_connection,
            'localhost',
            self.aggregator_port
        )
        
        print(f"[{self.agent_id}] ✓ Servidor agregador iniciado")
    
    async def connect_to_server(self):
        """Conecta con el servidor central"""
        print(f"[{self.agent_id}] Conectando al servidor {self.server_uri}...")
        
        async with websockets.connect(self.server_uri) as websocket:
            self.server_ws = websocket
            
            # Registrarse
            register_msg = {
                'type': 'register',
                'agent_id': self.agent_id
            }
            await websocket.send(json.dumps(register_msg))
            
            print(f"[{self.agent_id}] Esperando respuesta del servidor...")
            
            # Escuchar mensajes del servidor
            async for message in websocket:
                data = json.loads(message)
                await self.handle_server_message(data)
    
    async def handle_server_message(self, data):
        """Maneja mensajes del servidor central"""
        msg_type = data.get('type')
        
        if msg_type == 'registered':
            print(f"[{self.agent_id}] ✓ Registrado exitosamente en el servidor")
        
        elif msg_type == 'aggregator_selected':
            self.role = data.get('role')
            round_num = data.get('round')
            
            print(f"\n[{self.agent_id}] === RONDA {round_num + 1} - ROL: {self.role.upper()} ===")
            
            if self.role == 'aggregator':
                # Soy el agregador
                self.expected_agents = data.get('agents_list', [])
                print(f"[{self.agent_id}] Seleccionado como AGREGADOR")
                print(f"[{self.agent_id}] Esperando modelos de: {self.expected_agents}")
                
                # Reiniciar estado
                self.received_models = {}
                self.worker_connections = {}
                if hasattr(self, 'final_aggregated_params'):
                    delattr(self, 'final_aggregated_params')
                
                # Entrenar modelo local primero
                self.train_local_model(epochs=Config.LOCAL_EPOCHS)
                
                # Iniciar servidor para recibir modelos
                await self.start_aggregator_server()
                
            else:
                # Soy worker
                self.aggregator_info = data.get('aggregator_info')
                aggregator_id = data.get('aggregator_id')
                print(f"[{self.agent_id}] Rol: WORKER - Enviar modelo a {aggregator_id}")
                
                # Entrenar modelo local
                self.train_local_model(epochs=Config.LOCAL_EPOCHS)
                
                # Notificar al servidor que el entrenamiento terminó
                await self.server_ws.send(json.dumps({
                    'type': 'training_complete',
                    'agent_id': self.agent_id
                }))
                
                # Enviar modelo al agregador (con reintentos automáticos)
                await self.send_model_to_aggregator(self.aggregator_info)
                
                # NO confirmar aquí, esperar request_confirmation del servidor
                print(f"[{self.agent_id}] Esperando señal del servidor para siguiente ronda...")
        
        elif msg_type == 'request_confirmation':
            # El servidor solicita confirmación para siguiente ronda
            print(f"[{self.agent_id}] Recibida solicitud de confirmación del servidor")
            await asyncio.sleep(1)  # Pequeña pausa para asegurar que todo está listo
            await self.ready_for_next_round()
    
    async def ready_for_next_round(self):
        """Notifica al servidor que está listo para la siguiente ronda"""
        # Si fue agregador, cerrar el servidor
        if self.role == 'aggregator' and self.aggregator_server:
            self.aggregator_server.close()
            await self.aggregator_server.wait_closed()
            self.aggregator_server = None
            print(f"[{self.agent_id}] Servidor agregador cerrado")
        
        print(f"[{self.agent_id}] Listo para la siguiente ronda")
        if self.server_ws:
            message = {
                'type': 'ready_for_next_round',
                'agent_id': self.agent_id
            }
            await self.server_ws.send(json.dumps(message))
    
    async def start(self):
        """Inicia el agente"""
        print(f"\n{'='*60}")
        print(f"[{self.agent_id}] Iniciando agente...")
        print(f"{'='*60}\n")
        
        # Crear dataset local
        self.create_local_dataset()
        
        # Conectar al servidor
        await self.connect_to_server()


def main():
    if len(sys.argv) < 2:
        print("Uso: python agent.py <agent_id>")
        print("Ejemplo: python agent.py agent_1")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    agent = FederatedAgent(agent_id)
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print(f"\n[{agent_id}] Detenido por usuario")
    except Exception as e:
        print(f"[{agent_id}] Error: {e}")


if __name__ == '__main__':
    main()
