"""
Servidor Central para Aprendizaje Federado Semi-Descentralizado
Gestiona suscripciones de agentes y selecciona agregador mediante sorteo aleatorio
"""
import asyncio
import websockets
import json
import random
import torch
from typing import Dict, Set
from datetime import datetime
import csv
import os

# Importar configuración
# Para Raspberry Pi, cambiar a: from config import RaspberryPiConfig as Config
from config import RaspberryPiConfig as Config


class FederatedServer:
    def __init__(self, host='localhost', port=8765):
        self.host = host
        self.port = port
        self.agents: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.agents_info: Dict[str, dict] = {}
        self.current_round = 0
        self.waiting_for_agents = set()
        self.min_agents = Config.MIN_AGENTS
        self.early_stopping_patience = 4  # Early stopping en 4 rondas
        
        # Early stopping y métricas
        self.recall_history = []  # Histórico de recalls globales
        self.best_recall = 0.0
        self.prev_recall = 0.0
        self.rounds_without_improvement = 0
        self.training_stopped = False
        
        # Mejor modelo (para guardar al detener)
        self.best_model_recall = 0.0
        self.best_model_round = 0
        self.best_aggregator_id = None
        
        # Tracking para CSV
        self.current_aggregator_id = None
        self.num_local_train_completed = 0
        self.num_confirmations = 0
        
        # CSV logging
        self.server_log_file = "server_log.csv"
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
                self.current_aggregator_id,
                'lottery',
                Config.AGGREGATION_METHOD,
                len(self.agents),
                len(self.agents) - 1,  # num_workers
                self.num_local_train_completed,
                self.num_confirmations,
                1 if self.num_confirmations >= len(self.agents) else 0,
                f"{global_recall:.4f}",
                f"{self.best_recall:.4f}",
                f"{delta_recall:.4f}",
                1,  # early_stopping_enabled
                1 if early_stopped else 0,
                es_reason if early_stopped else '',
                self.early_stopping_patience,
                self.rounds_without_improvement,
                0 if early_stopped else 1
            ])
        
        self.prev_recall = global_recall
        self.num_local_train_completed = 0
        self.num_confirmations = 0
        
    async def register_agent(self, websocket, agent_id):
        """Registra un nuevo agente en el servidor"""
        self.agents[agent_id] = websocket
        
        # Obtener IP del agente desde la conexión WebSocket
        agent_ip = websocket.remote_address[0]
        
        self.agents_info[agent_id] = {
            'id': agent_id,
            'connected': True,
            'last_seen': datetime.now().isoformat(),
            'ip': agent_ip
        }
        print(f"[SERVIDOR] Agente {agent_id} registrado desde {agent_ip}. Total agentes: {len(self.agents)}")
        
    async def unregister_agent(self, agent_id):
        """Desregistra un agente del servidor"""
        if agent_id in self.agents:
            del self.agents[agent_id]
        if agent_id in self.agents_info:
            del self.agents_info[agent_id]
        print(f"[SERVIDOR] Agente {agent_id} desconectado. Total agentes: {len(self.agents)}")
        
    async def select_aggregator(self):
        """Selecciona un agregador mediante sorteo aleatorio (1-100)"""
        print(f"\n[SERVIDOR] === RONDA {self.current_round + 1} ===")
        print(f"[SERVIDOR] Realizando sorteo para seleccionar agregador...")
        
        # Cada agente obtiene un número aleatorio del 1 al 100
        lottery_results = {}
        for agent_id in self.agents.keys():
            lottery_number = random.randint(1, 100)
            lottery_results[agent_id] = lottery_number
            print(f"[SERVIDOR] Agente {agent_id} -> número {lottery_number}")
        
        # Seleccionar el ganador (número más alto)
        winner_id = max(lottery_results, key=lottery_results.get)
        winner_number = lottery_results[winner_id]
        
        print(f"[SERVIDOR] 🏆 Ganador: {winner_id} con número {winner_number}")
        
        self.current_aggregator_id = winner_id
        return winner_id, lottery_results
    
    async def broadcast_aggregator_info(self, aggregator_id, lottery_results):
        """Envía a todos los agentes quién es el agregador"""
        # Lista de otros agentes (sin el agregador)
        other_agents = [aid for aid in self.agents.keys() if aid != aggregator_id]
        
        # Obtener IP real del agregador
        aggregator_ip = self.agents_info[aggregator_id]['ip']
        
        # Si es IPv6 localhost (::1), convertir a IPv4 localhost
        if aggregator_ip in ['::1', '::ffff:127.0.0.1']:
            aggregator_ip = '127.0.0.1'
        
        # Información de conexión del agregador
        aggregator_info = {
            'id': aggregator_id,
            'host': aggregator_ip,  # IP real del agregador
            'port': Config.AGGREGATOR_PORT_BASE + int(aggregator_id.split('_')[1])
        }
        
        message_for_aggregator = {
            'type': 'aggregator_selected',
            'role': 'aggregator',
            'aggregator_id': aggregator_id,
            'lottery_results': lottery_results,
            'agents_list': other_agents,
            'round': self.current_round
        }
        
        message_for_workers = {
            'type': 'aggregator_selected',
            'role': 'worker',
            'aggregator_id': aggregator_id,
            'aggregator_info': aggregator_info,
            'lottery_results': lottery_results,
            'round': self.current_round
        }
        
        # Enviar mensajes
        for agent_id, ws in self.agents.items():
            try:
                if agent_id == aggregator_id:
                    await ws.send(json.dumps(message_for_aggregator))
                    print(f"[SERVIDOR] Notificado a {agent_id} como AGREGADOR")
                else:
                    await ws.send(json.dumps(message_for_workers))
                    print(f"[SERVIDOR] Notificado a {agent_id} como WORKER (enviar a {aggregator_id})")
            except Exception as e:
                print(f"[SERVIDOR] Error al notificar a {agent_id}: {e}")
    
    async def wait_for_all_agents(self):
        """Espera a que todos los agentes confirmen que están listos"""
        self.waiting_for_agents = set(self.agents.keys())
        print(f"[SERVIDOR] Esperando confirmación de {len(self.waiting_for_agents)} agentes...")
        
        # Enviar solicitud de confirmación
        confirm_message = {
            'type': 'request_confirmation',
            'message': 'Confirma cuando estés listo para la siguiente ronda'
        }
        
        for agent_id, ws in self.agents.items():
            try:
                await ws.send(json.dumps(confirm_message))
            except Exception as e:
                print(f"[SERVIDOR] Error al solicitar confirmación a {agent_id}: {e}")
    
    def agent_ready(self, agent_id):
        """Marca un agente como listo"""
        if agent_id in self.waiting_for_agents:
            self.waiting_for_agents.remove(agent_id)
            print(f"[SERVIDOR] Agente {agent_id} confirmado. Faltan {len(self.waiting_for_agents)} agentes")
            return len(self.waiting_for_agents) == 0
        return False
    
    def check_early_stopping(self, global_recall):
        """
        Verifica si se debe detener el entrenamiento
        
        Returns:
            tuple: (should_stop, reason)
        """
        # Guardar recall en histórico
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
            # Registrar mejor modelo info
            self.best_model_recall = global_recall
            self.best_model_round = self.current_round
            self.best_aggregator_id = self.current_aggregator_id
            print(f"[SERVIDOR] 💾 Mejor modelo registrado (Ronda {self.current_round}, Agregador {self.current_aggregator_id})")
        else:
            self.rounds_without_improvement += 1
            print(f"[SERVIDOR] ⚠️ Sin mejora: {self.rounds_without_improvement}/{self.early_stopping_patience} rondas")
        
        # Verificar early stopping
        if self.rounds_without_improvement >= self.early_stopping_patience:
            print(f"[SERVIDOR] ⛔ EARLY STOPPING: {self.early_stopping_patience} rondas sin mejora")
            return True, f"early_stopping_patience_{self.early_stopping_patience}"
        
        print(f"[SERVIDOR] ✓ Continuando entrenamiento")
        print(f"[SERVIDOR] =======================================\n")
        return False, None
    
    async def start_new_round(self):
        """Inicia una nueva ronda de entrenamiento federado"""
        if len(self.agents) < self.min_agents:
            print(f"[SERVIDOR] No hay suficientes agentes ({len(self.agents)}/{self.min_agents})")
            return
        
        # Seleccionar agregador
        aggregator_id, lottery_results = await self.select_aggregator()
        
        # Notificar a todos los agentes
        await self.broadcast_aggregator_info(aggregator_id, lottery_results)
        
        self.current_round += 1
    
    async def handle_agent(self, websocket, path):
        """Maneja la conexión de un agente"""
        agent_id = None
        try:
            async for message in websocket:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'register':
                    agent_id = data.get('agent_id')
                    await self.register_agent(websocket, agent_id)
                    
                    # Responder con confirmación
                    response = {
                        'type': 'registered',
                        'agent_id': agent_id,
                        'message': 'Registro exitoso'
                    }
                    await websocket.send(json.dumps(response))
                    
                    # Si hay suficientes agentes, iniciar ronda
                    if len(self.agents) >= self.min_agents and self.current_round == 0:
                        await asyncio.sleep(Config.STARTUP_DELAY)  # Dar tiempo para que se conecten más
                        await self.start_new_round()
                
                elif msg_type == 'ready_for_next_round':
                    agent_id = data.get('agent_id')
                    self.num_confirmations += 1
                    all_ready = self.agent_ready(agent_id)
                    
                    if all_ready:
                        if self.training_stopped:
                            print(f"[SERVIDOR] ⛔ Entrenamiento detenido. No se iniciarán más rondas.")
                        else:
                            print(f"[SERVIDOR] Todos los agentes listos. Iniciando nueva ronda...")
                            await asyncio.sleep(Config.ROUND_START_DELAY)
                            await self.start_new_round()
                
                elif msg_type == 'training_complete':
                    agent_id = data.get('agent_id')
                    self.num_local_train_completed += 1
                    print(f"[SERVIDOR] Agente {agent_id} completó entrenamiento local")
                
                elif msg_type == 'aggregation_complete':
                    agent_id = data.get('agent_id')
                    global_recall = data.get('global_recall', 0.0)
                    print(f"[SERVIDOR] Agregador {agent_id} completó agregación")
                    print(f"[SERVIDOR] Recall global reportado: {global_recall:.4f}")
                    
                    # Verificar early stopping
                    should_stop, reason = self.check_early_stopping(global_recall)
                    
                    # Guardar métricas en CSV
                    self.log_server_metrics(global_recall, should_stop, reason if should_stop else '')
                    
                    if should_stop:
                        self.training_stopped = True
                        print(f"\n{'='*70}")
                        print(f"[SERVIDOR] 🛑 ENTRENAMIENTO DETENIDO")
                        print(f"[SERVIDOR] Razón: {reason}")
                        print(f"[SERVIDOR] Total de rondas: {self.current_round}")
                        print(f"[SERVIDOR] Mejor recall alcanzado: {self.best_recall:.4f}")
                        print(f"[SERVIDOR] Mejor modelo en ronda: {self.best_model_round}")
                        print(f"[SERVIDOR] Agregador con mejor modelo: {self.best_aggregator_id}")
                        print(f"{'='*70}\n")
                        
                        # Solicitar al agregador que guarde el mejor modelo
                        if self.best_aggregator_id and self.best_aggregator_id in self.agents:
                            save_model_msg = {
                                'type': 'save_best_model',
                                'best_round': self.best_model_round,
                                'best_recall': self.best_model_recall
                            }
                            try:
                                await self.agents[self.best_aggregator_id].send(json.dumps(save_model_msg))
                                print(f"[SERVIDOR] 💾 Solicitado a {self.best_aggregator_id} que guarde el mejor modelo")
                            except:
                                print(f"[SERVIDOR] ⚠️ No se pudo solicitar guardar modelo a {self.best_aggregator_id}")
                        
                        # Notificar a todos los agentes que deben detenerse
                        stop_message = {
                            'type': 'training_stopped',
                            'reason': reason,
                            'total_rounds': self.current_round,
                            'best_recall': self.best_recall,
                            'best_model_round': self.best_model_round,
                            'recall_history': self.recall_history
                        }
                        for agent_ws in self.agents.values():
                            try:
                                await agent_ws.send(json.dumps(stop_message))
                            except:
                                pass
                    else:
                        # Solicitar confirmación para siguiente ronda
                        await self.wait_for_all_agents()
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"[SERVIDOR] Error manejando agente: {e}")
        finally:
            if agent_id:
                await self.unregister_agent(agent_id)
    
    async def start(self):
        """Inicia el servidor"""
        print(f"[SERVIDOR] Iniciando servidor en ws://{self.host}:{self.port}")
        print(f"[SERVIDOR] Esperando mínimo {self.min_agents} agentes para comenzar...")
        
        async with websockets.serve(self.handle_agent, self.host, self.port):
            await asyncio.Future()  # Mantener servidor corriendo indefinidamente


def main():
    server = FederatedServer(host='0.0.0.0', port=8765)
    asyncio.run(server.start())


if __name__ == '__main__':
    main()
