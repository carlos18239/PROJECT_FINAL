"""
Agente Worker para Aprendizaje Federado Centralizado
El agente solo actúa como worker, enviando modelos al servidor central
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
import joblib
import time
from datetime import datetime
import csv

from config import RaspberryPiConfig as Config
from mlp import MLP
from data_preparation import load_preprocessor


class CentralizedWorker:
    def __init__(self, agent_id, csv_file, server_uri='ws://localhost:8765'):
        self.agent_id = agent_id
        self.csv_file = csv_file
        self.server_uri = server_uri
        self.server_ws = None
        
        # Modelo y datos
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_data = None
        self.val_data = None
        self.n_features = None
        
        # Para FedProx: guardar modelo global
        self.global_model_params = None
        
        # Métricas y timing
        self.current_round = 0
        self.train_start_time = None
        
        # CSV logging
        self.worker_log_file = f"worker_{agent_id}_centralized_log.csv"
        self.init_csv_log()
        
        print(f"[{self.agent_id}] Inicializado con CSV: {csv_file}")
    
    def init_csv_log(self):
        """Inicializa archivo CSV para logging"""
        headers = [
            'timestamp', 'round', 'client_id', 'local_samples', 'local_epochs', 'local_batch_size',
            'local_train_time', 'local_send_time', 'local_receive_time',
            'local_upload_bytes', 'local_download_bytes', 'local_total_bytes',
            'local_loss', 'local_accuracy', 'local_precision', 'local_recall', 'local_f1', 'status'
        ]
        
        if not os.path.exists(self.worker_log_file):
            with open(self.worker_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
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
    
    def load_and_preprocess_data(self):
        """Carga y preprocesa datos desde CSV"""
        print(f"\n{'='*70}")
        print(f"[{self.agent_id}] INICIANDO AGENTE WORKER CENTRALIZADO")
        print(f"{'='*70}\n")
        
        print(f"[{self.agent_id}] Cargando datos desde {self.csv_file}...")
        
        # Cargar CSV
        df = pd.read_csv(self.csv_file)
        print(f"[{self.agent_id}] CSV cargado: {len(df)} filas")
        print(f"[{self.agent_id}] Columnas: {list(df.columns)}")
        
        # Cargar preprocessor
        preprocessor = load_preprocessor('preprocessor_global.joblib')
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
        
        # Separar target
        y = df[target_col].values
        
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
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Crear dataset
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
    
    def calculate_all_metrics(self):
        """Calcula todas las métricas"""
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
    
    async def connect_to_server(self):
        """Conecta con el servidor central"""
        print(f"[{self.agent_id}] Conectando al servidor {self.server_uri}...")
        
        async with websockets.connect(self.server_uri) as websocket:
            self.server_ws = websocket
            
            # Registrarse
            register_msg = {
                'type': 'register',
                'agent_id': self.agent_id,
                'n_features': self.n_features
            }
            await websocket.send(json.dumps(register_msg))
            
            print(f"[{self.agent_id}] Esperando respuesta del servidor...")
            
            # Escuchar mensajes del servidor
            async for message in websocket:
                data = json.loads(message)
                await self.handle_server_message(data, websocket)
    
    async def handle_server_message(self, data, websocket):
        """Maneja mensajes del servidor"""
        msg_type = data.get('type')
        
        if msg_type == 'registered':
            print(f"[{self.agent_id}] ✓ Registrado exitosamente en el servidor")
        
        elif msg_type == 'start_round':
            self.current_round = data.get('round')
            print(f"\n{'='*70}")
            print(f"[{self.agent_id}] === RONDA {self.current_round} ===")
            print(f"{'='*70}\n")
            
            # Recibir modelo global
            receive_start = time.time()
            model_bytes = await websocket.recv()
            receive_time = time.time() - receive_start
            download_bytes = len(model_bytes)
            
            global_params = pickle.loads(model_bytes)
            self.set_model_parameters(global_params)
            
            # Guardar modelo global para FedProx
            self.global_model_params = {name: param.clone() for name, param in global_params.items()}
            
            print(f"[{self.agent_id}] ✓ Modelo global recibido ({download_bytes} bytes)")
            
            # Entrenar localmente
            self.train_local_model()
            train_time = time.time() - self.train_start_time
            
            # Calcular métricas
            metrics = self.calculate_all_metrics()
            
            # Enviar modelo al servidor
            send_start = time.time()
            local_params = self.get_model_parameters()
            params_bytes = pickle.dumps(local_params)
            
            update_msg = {
                'type': 'model_update',
                'agent_id': self.agent_id,
                'local_recall': metrics['recall']
            }
            
            await websocket.send(json.dumps(update_msg))
            await websocket.send(params_bytes)
            
            send_time = time.time() - send_start
            upload_bytes = len(params_bytes)
            
            print(f"[{self.agent_id}] ✓ Modelo local enviado ({upload_bytes} bytes)")
            print(f"[{self.agent_id}] Recall local: {metrics['recall']:.4f}")
            
            # Evaluar
            self.evaluate_model()
            
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
                'local_loss': f"{metrics['loss']:.4f}",
                'local_accuracy': f"{metrics['accuracy']:.4f}",
                'local_precision': f"{metrics['precision']:.4f}",
                'local_recall': f"{metrics['recall']:.4f}",
                'local_f1': f"{metrics['f1']:.4f}",
                'status': 'OK'
            }
            
            self.log_worker_metrics(worker_metrics)
        
        elif msg_type == 'training_stopped':
            reason = data.get('reason', 'unknown')
            total_rounds = data.get('total_rounds', 0)
            best_recall = data.get('best_recall', 0.0)
            
            print(f"\n{'='*70}")
            print(f"[{self.agent_id}] 🛑 ENTRENAMIENTO DETENIDO POR SERVIDOR")
            print(f"[{self.agent_id}] Razón: {reason}")
            print(f"[{self.agent_id}] Total rondas: {total_rounds}")
            print(f"[{self.agent_id}] Mejor recall: {best_recall:.4f}")
            print(f"{'='*70}\n")
    
    async def start(self):
        """Inicia el agente"""
        # Cargar y preprocesar datos
        self.load_and_preprocess_data()
        
        # Conectar al servidor
        await self.connect_to_server()


def main():
    if len(sys.argv) < 3:
        print("Uso: python worker_centralized.py <agent_id> <csv_file> [server_uri]")
        print("Ejemplo local: python worker_centralized.py agent_1 data1.csv")
        print("Ejemplo red: python worker_centralized.py agent_1 data1.csv ws://192.168.1.100:8765")
        sys.exit(1)
    
    agent_id = sys.argv[1]
    csv_file = sys.argv[2]
    server_uri = sys.argv[3] if len(sys.argv) > 3 else 'ws://localhost:8765'
    
    worker = CentralizedWorker(agent_id, csv_file, server_uri)
    asyncio.run(worker.start())


if __name__ == '__main__':
    main()
