"""
Servidor Centralizado para Aprendizaje Federado
El servidor actúa como agregador central fijo en todas las rondas
"""
import asyncio
import websockets
import json
import torch
import pickle
from typing import Dict
from datetime import datetime
import csv
import os
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

from config import RaspberryPiConfig as Config
from mlp import MLP


class CentralizedServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.agents: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.current_round = 0
        self.min_agents = 3  # Reducido a 3 agentes
        
        # Modelo central (agregador)
        self.model = None
        self.n_features = None
        
        # Recepción de modelos
        self.received_models = {}
        self.received_recalls = {}
        
        # Early stopping y métricas
        self.recall_history = []
        self.best_recall = 0.0
        self.prev_recall = 0.0
        self.rounds_without_improvement = 0
        self.training_stopped = False
        
        # Mejor modelo (para guardar al detener)
        self.best_model_params = None
        self.best_model_round = 0
        
        # Tracking para CSV
        self.num_local_train_completed = 0
        self.num_confirmations = 0
        
        # CSV logging
        self.server_log_file = "server_centralized_log.csv"
        self.init_csv_log()
    
    def init_csv_log(self):
        """Inicializa archivo CSV para logging del servidor"""
        headers = [
            'timestamp', 'round', 'selected_aggregator_id', 'selection_strategy',
            'aggregation_method', 'num_agents_total', 'num_workers', 'num_local_train_completed',
            'num_confirmations', 'all_agents_confirmed', 'global_recall',
            'best_recall_so_far', 'delta_recall', 'early_stopping_enabled',
            'early_stopping_triggered', 'early_stopping_reason', 'patience_value',
            'patience_counter', 'continue_training'
        ]
        
        if not os.path.exists(self.server_log_file):
            with open(self.server_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_server_metrics(self, global_recall, early_stopped, es_reason):
        """Guarda métricas del servidor en CSV"""
        delta_recall = global_recall - self.prev_recall
        
        with open(self.server_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                self.current_round,
                'server',  # Servidor fijo como agregador
                'centralized',
                Config.AGGREGATION_METHOD,
                len(self.agents),
                len(self.agents),  # Todos son workers
                self.num_local_train_completed,
                self.num_confirmations,
                1 if self.num_confirmations >= len(self.agents) else 0,
                f"{global_recall:.4f}",
                f"{self.best_recall:.4f}",
                f"{delta_recall:.4f}",
                1,  # early_stopping_enabled
                1 if early_stopped else 0,
                es_reason if early_stopped else '',
                Config.EARLY_STOPPING_PATIENCE,
                self.rounds_without_improvement,
                0 if early_stopped else 1
            ])
        
        self.prev_recall = global_recall
        self.num_local_train_completed = 0
        self.num_confirmations = 0
    
    async def register_agent(self, websocket, agent_id):
        """Registra un nuevo agente"""
        self.agents[agent_id] = websocket
        print(f"[SERVIDOR] Agente {agent_id} registrado. Total agentes: {len(self.agents)}")
    
    async def unregister_agent(self, agent_id):
        """Desregistra un agente"""
        if agent_id in self.agents:
            del self.agents[agent_id]
        print(f"[SERVIDOR] Agente {agent_id} desconectado. Total agentes: {len(self.agents)}")
    
    def initialize_model(self, n_features):
        """Inicializa el modelo central"""
        if self.model is None:
            self.n_features = n_features
            self.model = MLP(in_features=n_features, seed=42, p_dropout=0.3)
            print(f"[SERVIDOR] Modelo central inicializado con {n_features} features")
    
    def get_model_parameters(self):
        """Obtiene los parámetros del modelo central"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters):
        """Establece los parámetros del modelo central"""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()
    
    def federated_averaging(self):
        """Implementa FedAvg para agregar modelos de workers"""
        print(f"[SERVIDOR] Ejecutando FedAvg con {len(self.received_models)} modelos...")
        
        all_models = list(self.received_models.values())
        aggregated_params = {}
        
        # Promediar cada parámetro
        for param_name in all_models[0].keys():
            param_sum = torch.zeros_like(all_models[0][param_name])
            for model_params in all_models:
                param_sum += model_params[param_name]
            
            aggregated_params[param_name] = param_sum / len(all_models)
        
        print(f"[SERVIDOR] ✓ FedAvg completado")
        return aggregated_params
    
    def save_best_model(self):
        """Guarda el mejor modelo en disco"""
        if self.best_model_params is None:
            print(f"[SERVIDOR] ⚠️ No hay mejor modelo para guardar")
            return
        
        # Crear directorio si no existe
        os.makedirs('models', exist_ok=True)
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"models/best_model_centralized_round{self.best_model_round}_recall{self.best_recall:.4f}_{timestamp}.pt"
        
        # Guardar modelo
        torch.save({
            'round': self.best_model_round,
            'recall': self.best_recall,
            'model_state_dict': self.best_model_params,
            'recall_history': self.recall_history
        }, filename)
        
        print(f"[SERVIDOR] 💾 Mejor modelo guardado: {filename}")
        print(f"[SERVIDOR] 📊 Ronda: {self.best_model_round}, Recall: {self.best_recall:.4f}")
        return filename
    
    def check_early_stopping(self, global_recall):
        """Verifica si se debe detener el entrenamiento"""
        self.recall_history.append({
            'round': self.current_round,
            'recall': global_recall
        })
        
        print(f"\n[SERVIDOR] ===== EVALUACIÓN EARLY STOPPING =====")
        print(f"[SERVIDOR] Ronda actual: {self.current_round}")
        print(f"[SERVIDOR] Recall global: {global_recall:.4f}")
        print(f"[SERVIDOR] Mejor recall: {self.best_recall:.4f}")
        
        # Verificar si alcanzamos el límite de rondas
        if self.current_round >= Config.MAX_ROUNDS:
            print(f"[SERVIDOR] ⛔ DETENIENDO: Alcanzado límite de {Config.MAX_ROUNDS} rondas")
            return True, f"max_rounds_{Config.MAX_ROUNDS}"
        
        # Verificar si hay mejora significativa
        if global_recall > self.best_recall + Config.MIN_RECALL_IMPROVEMENT:
            print(f"[SERVIDOR] ✅ MEJORA detectada: +{global_recall - self.best_recall:.4f}")
            self.best_recall = global_recall
            self.rounds_without_improvement = 0
            # Guardar mejor modelo
            self.best_model_params = self.get_model_parameters()
            self.best_model_round = self.current_round
            print(f"[SERVIDOR] 💾 Mejor modelo actualizado (Ronda {self.current_round})")
        else:
            self.rounds_without_improvement += 1
            print(f"[SERVIDOR] ⚠️ Sin mejora: {self.rounds_without_improvement}/{Config.EARLY_STOPPING_PATIENCE} rondas")
        
        # Verificar early stopping
        if self.rounds_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
            print(f"[SERVIDOR] ⛔ EARLY STOPPING: {Config.EARLY_STOPPING_PATIENCE} rondas sin mejora")
            return True, f"early_stopping_patience_{Config.EARLY_STOPPING_PATIENCE}"
        
        print(f"[SERVIDOR] ✓ Continuando entrenamiento")
        print(f"[SERVIDOR] =======================================\n")
        return False, None
    
    async def start_training_round(self):
        """Inicia una ronda de entrenamiento centralizado"""
        self.current_round += 1
        print(f"\n{'='*70}")
        print(f"[SERVIDOR] === RONDA {self.current_round} ===")
        print(f"{'='*70}\n")
        
        # Resetear estado
        self.received_models = {}
        self.received_recalls = {}
        self.num_local_train_completed = 0
        
        # Enviar modelo global a todos los workers
        model_params = self.get_model_parameters()
        params_bytes = pickle.dumps(model_params)
        
        start_round_msg = {
            'type': 'start_round',
            'round': self.current_round,
            'model_size': len(params_bytes)
        }
        
        # Enviar señal de inicio a todos
        for agent_id, ws in self.agents.items():
            try:
                await ws.send(json.dumps(start_round_msg))
                await ws.send(params_bytes)
                print(f"[SERVIDOR] Modelo global enviado a {agent_id}")
            except Exception as e:
                print(f"[SERVIDOR] Error enviando modelo a {agent_id}: {e}")
    
    async def handle_agent(self, websocket, path):
        """Maneja la conexión de un agente"""
        agent_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'register':
                    agent_id = data.get('agent_id')
                    n_features = data.get('n_features')
                    
                    await self.register_agent(websocket, agent_id)
                    
                    # Inicializar modelo si es el primer agente
                    if self.model is None:
                        self.initialize_model(n_features)
                    
                    # Responder con confirmación
                    response = {
                        'type': 'registered',
                        'agent_id': agent_id
                    }
                    await websocket.send(json.dumps(response))
                    
                    # Si tenemos suficientes agentes, iniciar primera ronda
                    if len(self.agents) >= self.min_agents and self.current_round == 0:
                        await asyncio.sleep(Config.STARTUP_DELAY)
                        await self.start_training_round()
                
                elif msg_type == 'model_update':
                    agent_id = data.get('agent_id')
                    local_recall = data.get('local_recall', 0.0)
                    
                    # Recibir modelo
                    model_bytes = await websocket.recv()
                    model_params = pickle.loads(model_bytes)
                    
                    self.received_models[agent_id] = model_params
                    self.received_recalls[agent_id] = local_recall
                    
                    print(f"[SERVIDOR] Modelo recibido de {agent_id} (Recall: {local_recall:.4f}) [{len(self.received_models)}/{len(self.agents)}]")
                    
                    # Si recibimos todos los modelos, agregar
                    if len(self.received_models) == len(self.agents):
                        print(f"\n[SERVIDOR] Todos los modelos recibidos. Ejecutando FedAvg...")
                        
                        aggregated_params = self.federated_averaging()
                        self.set_model_parameters(aggregated_params)
                        
                        # Calcular recall global
                        global_recall = sum(self.received_recalls.values()) / len(self.received_recalls)
                        print(f"[SERVIDOR] 📊 Recall Global: {global_recall:.4f}")
                        
                        # Verificar early stopping
                        should_stop, reason = self.check_early_stopping(global_recall)
                        
                        # Guardar métricas
                        self.log_server_metrics(global_recall, should_stop, reason if should_stop else '')
                        
                        if should_stop:
                            self.training_stopped = True
                            print(f"\n{'='*70}")
                            print(f"[SERVIDOR] 🛑 ENTRENAMIENTO DETENIDO")
                            print(f"[SERVIDOR] Razón: {reason}")
                            print(f"[SERVIDOR] Total de rondas: {self.current_round}")
                            print(f"[SERVIDOR] Mejor recall alcanzado: {self.best_recall:.4f}")
                            print(f"{'='*70}\n")
                            
                            # Guardar el mejor modelo
                            saved_file = self.save_best_model()
                            
                            # Notificar a todos los agentes
                            stop_message = {
                                'type': 'training_stopped',
                                'reason': reason,
                                'total_rounds': self.current_round,
                                'best_recall': self.best_recall,
                                'best_model_file': saved_file
                            }
                            
                            for agent_ws in self.agents.values():
                                try:
                                    await agent_ws.send(json.dumps(stop_message))
                                except:
                                    pass
                        else:
                            # Continuar con siguiente ronda
                            await asyncio.sleep(Config.ROUND_START_DELAY)
                            await self.start_training_round()
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[SERVIDOR] Error manejando agente: {e}")
        finally:
            if agent_id:
                await self.unregister_agent(agent_id)
    
    async def start(self):
        """Inicia el servidor"""
        print(f"[SERVIDOR] Iniciando servidor centralizado en ws://{self.host}:{self.port}")
        print(f"[SERVIDOR] Esperando mínimo {self.min_agents} agentes para comenzar...")
        
        async with websockets.serve(self.handle_agent, self.host, self.port):
            await asyncio.Future()


def main():
    server = CentralizedServer(host='0.0.0.0', port=8765)
    asyncio.run(server.start())


if __name__ == '__main__':
    main()
