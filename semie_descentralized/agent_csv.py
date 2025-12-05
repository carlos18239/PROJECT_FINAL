"""
Agente para Aprendizaje Federado Semi-Descentralizado con CSV Reales
Usa MLP, data_preparation y joblib para datos tabulares

Funcionalidad: 
- Selecciona un agregador mediante sorteo
- Realiza 5 RONDAS INTERNAS de entrena → envía → agrega → recibe
- Después de 5 rondas, vuelve al servidor para elegir nuevo agregador
"""
import asyncio
import websockets
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pickle
import sys
from sklearn.metrics import recall_score, precision_score, f1_score
import os
from typing import Dict, List
import joblib
import time
from datetime import datetime
import csv

# Importar módulos locales
from mlp import MLP
from data_preparation import run_preprocessing, get_default_config

# Importar configuración
from config import RaspberryPiConfig as Config


class FederatedAgentCSV:
    def __init__(self, agent_id, csv_file, server_uri='ws://localhost:8765'):
        self.agent_id = agent_id
        self.csv_file = csv_file
        self.server_uri = server_uri
        self.server_ws = None
        self.role = None  # 'worker' o 'aggregator'
        self.aggregator_info = None
        
        # Modelo y datos
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()  # Para clasificación binaria
        self.train_data = None
        self.val_data = None
        self.n_features = None
        
        # Para FedProx: guardar modelo global/agregado
        self.global_model_params = None
        
        # Para modo agregador
        self.aggregator_server = None
        self.received_models = {}
        self.expected_agents = []
        self.worker_connections = {}
        
        # Contador de rondas internas (5 rondas con el mismo agregador)
        self.internal_round_count = 0
        self.max_internal_rounds = Config.INTERNAL_ROUNDS
        self.current_aggregator_id = None
        
        # Puerto para recibir modelos cuando es agregador
        self.aggregator_port = Config.AGGREGATOR_PORT_BASE + int(agent_id.split('_')[1])
        
        # Métricas y timing para logging
        self.current_round = 0
        self.train_start_time = None
        self.send_start_time = None
        self.receive_start_time = None
        self.bytes_uploaded = 0
        self.bytes_downloaded = 0
        self.worker_metrics = []  # Para agregador: métricas de workers
        
        # CSV logging
        self.worker_log_file = f"worker_{agent_id}_log.csv"
        self.aggregator_log_file = f"aggregator_log.csv"
        self.init_csv_logs()
        
        print(f"[{self.agent_id}] Inicializado con CSV: {csv_file}")
    
    def load_and_preprocess_data(self):
        """Carga y preprocesa datos desde CSV usando data_preparation"""
        print(f"[{self.agent_id}] Cargando datos desde {self.csv_file}...")
        
        # Configurar preprocesamiento
        base_dir = os.path.dirname(os.path.abspath(__file__))
        preprocessor_path = os.path.join(base_dir, "preprocessor_global.joblib")
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV no encontrado: {self.csv_file}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor no encontrado: {preprocessor_path}")
        
        # Leer CSV crudo
        df = pd.read_csv(self.csv_file)
        print(f"[{self.agent_id}] CSV cargado: {len(df)} filas")
        print(f"[{self.agent_id}] Columnas: {list(df.columns)}")
        
        # Cargar preprocessor
        preprocessor = joblib.load(preprocessor_path)
        print(f"[{self.agent_id}] Preprocessor cargado")
        
        # Detectar columna target (priorizar is_premature_ncd)
        target_col = None
        for possible_name in ['is_premature_ncd', 'Classification', 'target', 'label', 'y']:
            if possible_name in df.columns:
                target_col = possible_name
                break
        
        # Si no encuentra ninguna, usar la última columna EXCEPTO hospital_cliente
        if target_col is None:
            last_col = df.columns[-1]
            if last_col != 'hospital_cliente':
                target_col = last_col
            else:
                # Si la última es hospital_cliente, buscar otra columna numérica binaria
                for col in df.columns:
                    if col != 'hospital_cliente' and df[col].nunique() == 2:
                        target_col = col
                        break
        
        if target_col is None:
            raise ValueError(f"No se pudo detectar columna target en: {list(df.columns)}")
        
        print(f"[{self.agent_id}] Columna target detectada: {target_col}")
        
        # Separar features y target
        y = df[target_col].astype(float).to_numpy()
        
        # Eliminar target, hospital_cliente y ncd_group de features
        cols_to_drop = [target_col]
        if 'hospital_cliente' in df.columns and target_col != 'hospital_cliente':
            cols_to_drop.append('hospital_cliente')
            print(f"[{self.agent_id}] Eliminando columna: hospital_cliente")
        if 'ncd_group' in df.columns and target_col != 'ncd_group':
            cols_to_drop.append('ncd_group')
            print(f"[{self.agent_id}] Eliminando columna: ncd_group")
        
        X = df.drop(cols_to_drop, axis=1)  # Mantener como DataFrame
        
        # Transformar features (ColumnTransformer espera DataFrame)
        X_transformed = preprocessor.transform(X)
        self.n_features = X_transformed.shape[1]
        
        print(f"[{self.agent_id}] Features transformadas: {self.n_features} columnas")
        print(f"[{self.agent_id}] Total muestras: {len(y)}")
        print(f"[{self.agent_id}] Distribución target: {np.bincount(y.astype(int))}")
        
        # Convertir a tensores
        X_tensor = torch.FloatTensor(X_transformed)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Shape: [N, 1]
        
        # Crear dataset y dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.train_data = DataLoader(
            train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True
        )
        self.val_data = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False
        )
        
        print(f"[{self.agent_id}] Train: {train_size} muestras, Val: {val_size} muestras")
        
        # Crear modelo
        self.model = MLP(in_features=self.n_features, seed=42, p_dropout=0.3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        print(f"[{self.agent_id}] Modelo MLP creado con {self.n_features} features")
    
    def init_csv_logs(self):
        """Inicializa archivos CSV para logging"""
        # CSV para worker (este agente cuando actúa como worker)
        worker_headers = [
            'timestamp', 'round', 'client_id', 'local_samples', 'local_epochs', 'local_batch_size',
            'local_train_time', 'local_send_time', 'local_receive_time',
            'local_upload_bytes', 'local_download_bytes', 'local_total_bytes',
            'local_loss', 'local_accuracy', 'local_precision', 'local_recall', 'local_f1', 'status'
        ]
        
        if not os.path.exists(self.worker_log_file):
            with open(self.worker_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(worker_headers)
        
        # CSV para agregador (solo se crea si este agente es seleccionado como agregador)
        aggregator_headers = [
            'timestamp', 'round', 'aggregator_id', 'num_clients_total', 'num_clients_participating',
            'num_clients_failed', 'samples_global', 'global_loss', 'global_accuracy',
            'global_precision', 'global_recall', 'global_f1', 'local_loss_mean',
            'local_accuracy_mean', 'local_recall_mean', 'local_f1_mean', 'local_accuracy_std',
            'num_messages', 'bytes_upload_round', 'bytes_download_round',
            'bytes_round_total', 'bytes_cumulative', 'train_time_mean',
            'aggregation_time', 'latency_wait_global', 'round_time'
        ]
        
        if not os.path.exists(self.aggregator_log_file):
            with open(self.aggregator_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(aggregator_headers)
    
    def log_worker_metrics(self, metrics):
        """Guarda métricas del worker en CSV"""
        with open(self.worker_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['timestamp'], metrics['round'], metrics['client_id'],
                metrics['local_samples'], metrics['local_epochs'], metrics['local_batch_size'],
                metrics['local_train_time'], metrics['local_send_time'], metrics['local_receive_time'],
                metrics['local_upload_bytes'], metrics['local_download_bytes'], metrics['local_total_bytes'],
                metrics['local_loss'], metrics['local_accuracy'], metrics['local_precision'],
                metrics['local_recall'], metrics['local_f1'], metrics['status']
            ])
    
    def log_aggregator_metrics(self, metrics):
        """Guarda métricas del agregador en CSV"""
        with open(self.aggregator_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics['timestamp'], metrics['round'], metrics['aggregator_id'],
                metrics['num_clients_total'], metrics['num_clients_participating'],
                metrics['num_clients_failed'], metrics['samples_global'], metrics['global_loss'],
                metrics['global_accuracy'], metrics['global_precision'], metrics['global_recall'],
                metrics['global_f1'], metrics['local_loss_mean'], metrics['local_accuracy_mean'],
                metrics['local_recall_mean'], metrics['local_f1_mean'], metrics['local_accuracy_std'],
                metrics['num_messages'], metrics['bytes_upload_round'], metrics['bytes_download_round'],
                metrics['bytes_round_total'], metrics['bytes_cumulative'], metrics['train_time_mean'],
                metrics['aggregation_time'], metrics['latency_wait_global'], metrics['round_time']
            ])
    
    def train_local_model(self, epochs=None):
        """Entrena el modelo con datos locales (soporta FedAvg y FedProx)"""
        if epochs is None:
            epochs = Config.LOCAL_EPOCHS
        
        self.train_start_time = time.time()
        method = Config.AGGREGATION_METHOD
        print(f"[{self.agent_id}] Iniciando entrenamiento local ({epochs} épocas, {method})...")
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in self.train_data:
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # FedProx: Agregar término proximal
                if method == 'FedProx' and self.global_model_params is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        proximal_term += ((param - self.global_model_params[name]) ** 2).sum()
                    loss += (Config.FEDPROX_MU / 2) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calcular accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
            
            avg_loss = total_loss / len(self.train_data)
            accuracy = 100 * correct / total
            print(f"[{self.agent_id}] Época {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        print(f"[{self.agent_id}] ✓ Entrenamiento local completado")
    
    def evaluate_model(self):
        """Evalúa el modelo en datos de validación"""
        if self.val_data is None:
            return
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_data:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(self.val_data)
        accuracy = 100 * correct / total
        print(f"[{self.agent_id}] Validación - Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
    
    def calculate_recall(self):
        """Calcula el recall local en datos de validación"""
        if self.val_data is None:
            return 0.0
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_data:
                outputs = self.model(batch_x)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(batch_y.cpu().numpy().flatten())
        
        # Usar 'macro' para promediar recall de ambas clases (0 y 1)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        return recall
    
    def calculate_all_metrics(self):
        """Calcula todas las métricas (loss, accuracy, precision, recall, f1)"""
        if self.val_data is None:
            return {
                'loss': 0.0, 'accuracy': 0.0, 'precision': 0.0,
                'recall': 0.0, 'f1': 0.0
            }
        
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_data:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(batch_y.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(self.val_data)
        accuracy = sum([1 for p, l in zip(all_preds, all_labels) if p == l]) / len(all_labels)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def get_model_parameters(self):
        """Obtiene los parámetros del modelo"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_model_parameters(self, parameters):
        """Establece los parámetros del modelo"""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()
    
    def save_best_model(self, best_round, best_recall):
        """Guarda el mejor modelo en disco"""
        # Crear directorio si no existe
        os.makedirs('models', exist_ok=True)
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"models/best_model_semidesc_{self.agent_id}_round{best_round}_recall{best_recall:.4f}_{timestamp}.pt"
        
        # Guardar modelo
        torch.save({
            'agent_id': self.agent_id,
            'round': best_round,
            'recall': best_recall,
            'model_state_dict': self.get_model_parameters() if hasattr(self, 'final_aggregated_params') == False else self.final_aggregated_params,
            'timestamp': timestamp
        }, filename)
        
        print(f"[{self.agent_id}] 💾 Mejor modelo guardado: {filename}")
        print(f"[{self.agent_id}] 📊 Ronda: {best_round}, Recall: {best_recall:.4f}")
        return filename
    
    async def send_model_to_aggregator(self, aggregator_info, round_num):
        """Envía el modelo local y recall al agregador"""
        aggregator_uri = f"ws://{aggregator_info['host']}:{aggregator_info['port']}"
        max_retries = Config.MAX_RETRIES
        retry_delay = Config.RETRY_DELAY
        
        # Calcular métricas locales antes de enviar
        metrics_before = self.calculate_all_metrics()
        
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"[{self.agent_id}] Conectando con agregador...")
                else:
                    print(f"[{self.agent_id}] Reintento {attempt}/{max_retries}...")
                
                send_start = time.time()
                
                async with websockets.connect(aggregator_uri) as ws:
                    # Serializar parámetros del modelo
                    model_params = self.get_model_parameters()
                    params_bytes = pickle.dumps(model_params)
                    
                    message = {
                        'type': 'model_update',
                        'agent_id': self.agent_id,
                        'params_size': len(params_bytes),
                        'internal_round': round_num,
                        'local_recall': metrics_before['recall']
                    }
                    
                    # Enviar metadata
                    await ws.send(json.dumps(message))
                    
                    # Enviar parámetros
                    await ws.send(params_bytes)
                    
                    send_time = time.time() - send_start
                    upload_bytes = len(params_bytes)
                    
                    print(f"[{self.agent_id}] ✓ Modelo enviado ({len(params_bytes)} bytes)")
                    print(f"[{self.agent_id}] Recall local: {metrics_before['recall']:.4f}")
                    
                    # Esperar modelo agregado
                    print(f"[{self.agent_id}] Esperando modelo agregado...")
                    receive_start = time.time()
                    response = await ws.recv()
                    
                    if isinstance(response, bytes):
                        download_bytes = len(response)
                        aggregated_params = pickle.loads(response)
                        self.set_model_parameters(aggregated_params)
                        
                        # Guardar modelo agregado para FedProx
                        self.global_model_params = {name: param.clone() for name, param in aggregated_params.items()}
                        
                        receive_time = time.time() - receive_start
                        
                        print(f"[{self.agent_id}] ✓ Modelo agregado recibido")
                        
                        # Evaluar modelo actualizado
                        self.evaluate_model()
                        
                        # Calcular tiempo de entrenamiento
                        train_time = time.time() - self.train_start_time if self.train_start_time else 0
                        
                        # Guardar métricas en CSV
                        train_size = len(self.train_data.dataset) if self.train_data else 0
                        
                        worker_metrics = {
                            'timestamp': datetime.now().isoformat(),
                            'round': self.current_round,
                            'client_id': self.agent_id,
                            'local_samples': train_size,
                            'local_epochs': Config.LOCAL_EPOCHS,
                            'local_batch_size': Config.BATCH_SIZE,
                            'local_train_time': f"{train_time:.4f}",
                            'local_send_time': f"{send_time:.4f}",
                            'local_receive_time': f"{receive_time:.4f}",
                            'local_upload_bytes': upload_bytes,
                            'local_download_bytes': download_bytes,
                            'local_total_bytes': upload_bytes + download_bytes,
                            'local_loss': f"{metrics_before['loss']:.4f}",
                            'local_accuracy': f"{metrics_before['accuracy']:.4f}",
                            'local_precision': f"{metrics_before['precision']:.4f}",
                            'local_recall': f"{metrics_before['recall']:.4f}",
                            'local_f1': f"{metrics_before['f1']:.4f}",
                            'status': 'OK'
                        }
                        
                        self.log_worker_metrics(worker_metrics)
                    
                    return True  # Éxito
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"[{self.agent_id}] Error después de {max_retries} intentos: {e}")
                    # Log como FAILED
                    worker_metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'round': self.current_round,
                        'client_id': self.agent_id,
                        'local_samples': 0,
                        'local_epochs': Config.LOCAL_EPOCHS,
                        'local_batch_size': Config.BATCH_SIZE,
                        'local_train_time': '0',
                        'local_send_time': '0',
                        'local_receive_time': '0',
                        'local_upload_bytes': 0,
                        'local_download_bytes': 0,
                        'local_total_bytes': 0,
                        'local_loss': '0',
                        'local_accuracy': '0',
                        'local_precision': '0',
                        'local_recall': '0',
                        'local_f1': '0',
                        'status': 'FAILED'
                    }
                    self.log_worker_metrics(worker_metrics)
                    return False
        
        return False
    
    async def handle_worker_connection(self, websocket, path):
        """Maneja conexiones de workers cuando este agente es agregador"""
        worker_id = None
        try:
            # Recibir metadata
            metadata = await websocket.recv()
            data = json.loads(metadata)
            
            if data['type'] == 'model_update':
                worker_id = data['agent_id']
                round_num = data.get('internal_round', 1)
                local_recall = data.get('local_recall', 0.0)
                print(f"[{self.agent_id}] Recibiendo modelo de {worker_id} (Ronda {round_num})...")
                print(f"[{self.agent_id}] Recall local de {worker_id}: {local_recall:.4f}")
                
                # Recibir parámetros
                params_bytes = await websocket.recv()
                model_params = pickle.loads(params_bytes)
                
                self.received_models[worker_id] = model_params
                
                # Guardar recall local
                if not hasattr(self, 'received_recalls'):
                    self.received_recalls = {}
                self.received_recalls[worker_id] = local_recall
                
                print(f"[{self.agent_id}] ✓ Modelo de {worker_id} recibido ({len(self.received_models)}/{len(self.expected_agents)})")
                
                # Esperar a que lleguen TODOS los modelos de esta ronda
                while len(self.received_models) < len(self.expected_agents):
                    await asyncio.sleep(0.5)
                
                # Agregar solo cuando llega el último modelo de esta ronda (solo una vez)
                if not self.aggregated_this_round:
                    self.aggregated_this_round = True
                    self.internal_round_count += 1
                    print(f"\n[{self.agent_id}] ===== RONDA INTERNA {self.internal_round_count}/{self.max_internal_rounds} =====")
                    print(f"[{self.agent_id}] Todos los modelos recibidos. Ejecutando FedAvg...")
                    
                    aggregated_params = self.federated_averaging()
                    
                    # Actualizar modelo propio
                    self.set_model_parameters(aggregated_params)
                    
                    # Evaluar modelo agregado
                    print(f"[{self.agent_id}] Evaluando modelo agregado:")
                    self.evaluate_model()
                    
                    # Calcular métricas del agregador para ESTA ronda interna
                    if hasattr(self, 'received_recalls') and self.received_recalls:
                        global_recall = sum(self.received_recalls.values()) / len(self.received_recalls)
                    else:
                        global_recall = 0.0
                    
                    agg_metrics = self.calculate_all_metrics()
                    
                    # Guardar métricas del agregador en CSV (cada ronda interna)
                    aggregator_metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'round': f"{self.current_round}.{self.internal_round_count}",  # Ej: 1.1, 1.2, etc.
                        'aggregator_id': self.agent_id,
                        'num_clients_total': len(self.expected_agents),
                        'num_clients_participating': len(self.received_models),
                        'num_clients_failed': 0,
                        'samples_global': 0,
                        'global_loss': f"{agg_metrics['loss']:.4f}",
                        'global_accuracy': f"{agg_metrics['accuracy']:.4f}",
                        'global_precision': f"{agg_metrics['precision']:.4f}",
                        'global_recall': f"{global_recall:.4f}",
                        'global_f1': f"{agg_metrics['f1']:.4f}",
                        'local_loss_mean': '0',
                        'local_accuracy_mean': '0',
                        'local_recall_mean': f"{global_recall:.4f}",
                        'local_f1_mean': '0',
                        'local_accuracy_std': '0',
                        'num_messages': len(self.received_models) * 2,
                        'bytes_upload_round': 0,
                        'bytes_download_round': 0,
                        'bytes_round_total': 0,
                        'bytes_cumulative': 0,
                        'train_time_mean': '0',
                        'aggregation_time': '0',
                        'latency_wait_global': '0',
                        'round_time': '0'
                    }
                    
                    self.log_aggregator_metrics(aggregator_metrics)
                    print(f"[{self.agent_id}] 📊 Métricas guardadas - Ronda interna {self.internal_round_count}")
                    
                    # Guardar modelo agregado para distribución
                    self.final_aggregated_params = aggregated_params
                    print(f"[{self.agent_id}] Modelo agregado listo para distribución")
                
                # Enviar modelo agregado a este worker
                params_bytes = pickle.dumps(self.final_aggregated_params)
                await websocket.send(params_bytes)
                print(f"[{self.agent_id}] ✓ Modelo agregado enviado a {worker_id}")
                
                self.models_sent_count += 1
                
                # Si ya enviamos a todos los workers, preparar siguiente ronda
                if self.models_sent_count >= len(self.expected_agents):
                    if self.internal_round_count >= self.max_internal_rounds:
                        print(f"\n[{self.agent_id}] ✅ COMPLETADAS {self.max_internal_rounds} RONDAS INTERNAS")
                        
                        # Calcular recall global final
                        if hasattr(self, 'received_recalls') and self.received_recalls:
                            global_recall = sum(self.received_recalls.values()) / len(self.received_recalls)
                            print(f"[{self.agent_id}] 📊 Recall Global Final: {global_recall:.4f}")
                            print(f"[{self.agent_id}] Recalls locales: {self.received_recalls}")
                        else:
                            global_recall = 0.0
                            print(f"[{self.agent_id}] ⚠️ No hay recalls locales disponibles")
                        
                        print(f"[{self.agent_id}] Notificando al servidor...")
                        await self.notify_server_aggregation_complete(global_recall)
                        # NO llamar ready_for_next_round aquí, esperar request_confirmation del servidor
                    else:
                        # Preparar para siguiente ronda interna
                        print(f"[{self.agent_id}] Preparando ronda {self.internal_round_count + 1}...")
                        print(f"[{self.agent_id}] Agregador listo para siguiente ronda (NO entrena)")
                        
                        # Resetear para siguiente ronda (NO resetear final_aggregated_params)
                        self.received_models = {}
                        self.received_recalls = {}
                        self.aggregated_this_round = False
                        self.models_sent_count = 0
                    
        except Exception as e:
            print(f"[{self.agent_id}] Error manejando worker {worker_id}: {e}")
    
    def federated_averaging(self):
        """Implementa FedAvg para agregar modelos"""
        print(f"[{self.agent_id}] Ejecutando FedAvg con {len(self.received_models)} modelos (solo workers)...")
        
        # SOLO usar modelos de workers (agregador NO participa con su modelo)
        all_models = list(self.received_models.values())
        
        # Inicializar parámetros agregados
        aggregated_params = {}
        
        # Promediar cada parámetro
        for param_name in all_models[0].keys():
            param_sum = torch.zeros_like(all_models[0][param_name])
            for model_params in all_models:
                param_sum += model_params[param_name]
            
            aggregated_params[param_name] = param_sum / len(all_models)
        
        print(f"[{self.agent_id}] ✓ FedAvg completado")
        self.final_aggregated_params = aggregated_params
        return aggregated_params
    
    async def distribute_aggregated_model(self, aggregated_params):
        """Distribuye el modelo agregado a todos los workers"""
        print(f"[{self.agent_id}] Modelo agregado listo para distribución")
    
    async def notify_server_aggregation_complete(self, global_recall=0.0):
        """Notifica al servidor que completó todas las rondas internas con el recall global"""
        if self.server_ws:
            message = {
                'type': 'aggregation_complete',
                'agent_id': self.agent_id,
                'internal_rounds_completed': self.internal_round_count,
                'global_recall': global_recall
            }
            await self.server_ws.send(json.dumps(message))
    
    async def start_aggregator_server(self):
        """Inicia servidor para recibir modelos de workers"""
        if self.aggregator_server:
            self.aggregator_server.close()
            await self.aggregator_server.wait_closed()
            print(f"[{self.agent_id}] Servidor agregador anterior cerrado")
        
        print(f"[{self.agent_id}] Iniciando servidor agregador en puerto {self.aggregator_port}...")
        
        # Escuchar en todas las interfaces (0.0.0.0) para red distribuida
        self.aggregator_server = await websockets.serve(
            self.handle_worker_connection,
            '0.0.0.0',
            self.aggregator_port
        )
        
        print(f"[{self.agent_id}] ✓ Servidor agregador iniciado en 0.0.0.0:{self.aggregator_port}")
    
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
            self.current_round = round_num + 1  # Servidor envía round después de incrementar
            
            print(f"\n{'='*70}")
            print(f"[{self.agent_id}] === SELECCIÓN RONDA {self.current_round} - ROL: {self.role.upper()} ===")
            print(f"{'='*70}\n")
            
            if self.role == 'aggregator':
                # Soy el agregador
                self.expected_agents = data.get('agents_list', [])
                self.current_aggregator_id = self.agent_id
                print(f"[{self.agent_id}] 🏆 Seleccionado como AGREGADOR")
                print(f"[{self.agent_id}] Esperando modelos de: {self.expected_agents}")
                print(f"[{self.agent_id}] Realizará {self.max_internal_rounds} rondas internas de agregación")
                
                # Reiniciar estado
                self.received_models = {}
                self.received_recalls = {}
                self.worker_connections = {}
                self.internal_round_count = 0
                self.aggregated_this_round = False
                self.models_sent_count = 0
                if hasattr(self, 'final_aggregated_params'):
                    delattr(self, 'final_aggregated_params')
                
                # El agregador NO entrena, solo hace FedAvg
                print(f"\n[{self.agent_id}] Preparando ronda interna 1...")
                print(f"[{self.agent_id}] Agregador listo (NO entrena, solo agrega)")
                
                # Iniciar servidor para recibir modelos
                await self.start_aggregator_server()
                
            else:
                # Soy worker
                self.aggregator_info = data.get('aggregator_info')
                aggregator_id = data.get('aggregator_id')
                self.current_aggregator_id = aggregator_id
                self.internal_round_count = 0
                
                print(f"[{self.agent_id}] Rol: WORKER")
                print(f"[{self.agent_id}] Agregador asignado: {aggregator_id}")
                print(f"[{self.agent_id}] Participará en {self.max_internal_rounds} rondas internas")
                
                # Notificar al servidor
                await self.server_ws.send(json.dumps({
                    'type': 'training_complete',
                    'agent_id': self.agent_id
                }))
                
                # Ejecutar 5 rondas internas en tarea separada
                asyncio.create_task(self.worker_internal_rounds_loop())
        
        elif msg_type == 'request_confirmation':
            # El servidor solicita confirmación para siguiente ronda
            print(f"[{self.agent_id}] Recibida solicitud de confirmación del servidor")
            await asyncio.sleep(1)
            await self.ready_for_next_round()
        
        elif msg_type == 'save_best_model':
            # Servidor solicita guardar el mejor modelo (solo para agregador)
            best_round = data.get('best_round', 0)
            best_recall = data.get('best_recall', 0.0)
            
            print(f"\n{'='*70}")
            print(f"[{self.agent_id}] 💾 GUARDANDO MEJOR MODELO")
            print(f"[{self.agent_id}] Ronda: {best_round}")
            print(f"[{self.agent_id}] Recall: {best_recall:.4f}")
            print(f"{'='*70}\n")
            
            self.save_best_model(best_round, best_recall)
        
        elif msg_type == 'training_stopped':
            # El servidor detuvo el entrenamiento (early stopping o max rounds)
            reason = data.get('reason', 'unknown')
            total_rounds = data.get('total_rounds', 0)
            best_recall = data.get('best_recall', 0.0)
            
            print(f"\n{'='*70}")
            print(f"[{self.agent_id}] 🛑 ENTRENAMIENTO DETENIDO POR SERVIDOR")
            print(f"[{self.agent_id}] Razón: {reason}")
            print(f"[{self.agent_id}] Total rondas: {total_rounds}")
            print(f"[{self.agent_id}] Mejor recall: {best_recall:.4f}")
            print(f"{'='*70}\n")
            
            # Cerrar servidor agregador si existe
            if self.aggregator_server:
                self.aggregator_server.close()
                await self.aggregator_server.wait_closed()
                print(f"[{self.agent_id}] Servidor agregador cerrado")
    
    async def worker_internal_rounds_loop(self):
        """Loop de 5 rondas internas para workers: entrena → envía → recibe"""
        for round_num in range(1, self.max_internal_rounds + 1):
            print(f"\n{'='*70}")
            print(f"[{self.agent_id}] === RONDA INTERNA {round_num}/{self.max_internal_rounds} ===")
            print(f"{'='*70}")
            
            # Entrenar modelo local
            self.train_local_model(epochs=Config.LOCAL_EPOCHS)
            
            # Enviar modelo al agregador y recibir modelo agregado
            success = await self.send_model_to_aggregator(self.aggregator_info, round_num)
            
            if not success:
                print(f"[{self.agent_id}] ⚠️ Error en ronda interna {round_num}")
                break
            
            # Pequeña pausa entre rondas (excepto en la última)
            if round_num < self.max_internal_rounds:
                print(f"[{self.agent_id}] Pausando 3 segundos antes de siguiente ronda...")
                await asyncio.sleep(3)
        
        # Completadas todas las rondas internas
        print(f"\n{'='*70}")
        print(f"[{self.agent_id}] ✅ COMPLETADAS {self.max_internal_rounds} RONDAS INTERNAS")
        print(f"[{self.agent_id}] Notificando al servidor...")
        print(f"{'='*70}\n")
        
        # Notificar al servidor que estamos listos para la siguiente ronda
        await self.ready_for_next_round()
    
    async def ready_for_next_round(self):
        """Notifica al servidor que está listo para la siguiente ronda"""
        # Si fue agregador, cerrar el servidor
        if self.role == 'aggregator' and self.aggregator_server:
            self.aggregator_server.close()
            await self.aggregator_server.wait_closed()
            self.aggregator_server = None
            print(f"[{self.agent_id}] Servidor agregador cerrado")
        
        print(f"[{self.agent_id}] Listo para nueva selección de agregador")
        if self.server_ws:
            message = {
                'type': 'ready_for_next_round',
                'agent_id': self.agent_id
            }
            await self.server_ws.send(json.dumps(message))
    
    async def start(self):
        """Inicia el agente"""
        print(f"\n{'='*70}")
        print(f"[{self.agent_id}] INICIANDO AGENTE CON DATOS REALES")
        print(f"{'='*70}\n")
        
        # Cargar y preprocesar datos
        self.load_and_preprocess_data()
        
        # Conectar al servidor
        await self.connect_to_server()


def main():
    if len(sys.argv) < 3:
        print("Uso: python agent_csv.py <agent_id> <csv_file> [server_uri]")
        print("Ejemplo local: python agent_csv.py agent_1 data1.csv")
        print("Ejemplo red: python agent_csv.py agent_1 data1.csv ws://192.168.1.100:8765")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    csv_file = sys.argv[2]
    server_uri = sys.argv[3] if len(sys.argv) > 3 else 'ws://localhost:8765'
    
    agent = FederatedAgentCSV(agent_id, csv_file, server_uri)
    
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print(f"\n[{agent_id}] Detenido por usuario")
    except Exception as e:
        print(f"[{agent_id}] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
