"""
Script de Análisis Comparativo para Aprendizaje Federado
Genera gráficas comparativas de métricas entre FedAvg y FedProx
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime


class FederatedAnalyzer:
    def __init__(self, results_dir='.'):
        self.results_dir = results_dir
        self.server_data = None
        self.worker_data = {}
        self.aggregator_data = None
        
    def load_data(self, prefix=''):
        """Carga todos los archivos CSV"""
        print(f"Cargando datos desde {self.results_dir}...")
        
        # Cargar datos del servidor
        server_files = glob.glob(os.path.join(self.results_dir, f'{prefix}server*log.csv'))
        if server_files:
            self.server_data = pd.read_csv(server_files[0])
            print(f"✓ Servidor: {len(self.server_data)} rondas")
        
        # Cargar datos de workers
        worker_files = glob.glob(os.path.join(self.results_dir, f'{prefix}worker*log.csv'))
        for wf in worker_files:
            agent_id = os.path.basename(wf).split('_')[1]
            self.worker_data[agent_id] = pd.read_csv(wf)
            print(f"✓ Worker {agent_id}: {len(self.worker_data[agent_id])} rondas")
        
        # Cargar datos de agregador (solo semi-descentralizado)
        agg_files = glob.glob(os.path.join(self.results_dir, f'{prefix}aggregator*log.csv'))
        if agg_files:
            self.aggregator_data = pd.read_csv(agg_files[0])
            print(f"✓ Agregador: {len(self.aggregator_data)} rondas")
    
    def plot_convergence_comparison(self, methods=['FedAvg', 'FedProx'], save_path='convergence_comparison.png'):
        """Gráfica de convergencia: Recall vs Rondas"""
        if self.server_data is None:
            print("⚠️ No hay datos del servidor")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparación de Convergencia: FedAvg vs FedProx', fontsize=16, fontweight='bold')
        
        # 1. Recall Global
        ax = axes[0, 0]
        for method in methods:
            method_data = self.server_data[self.server_data['aggregation_method'] == method]
            if len(method_data) > 0:
                ax.plot(method_data['round'], method_data['global_recall'], 
                       marker='o', label=method, linewidth=2, markersize=4)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Recall Global', fontsize=12)
        ax.set_title('Recall Global por Ronda', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Mejor Recall Acumulado
        ax = axes[0, 1]
        for method in methods:
            method_data = self.server_data[self.server_data['aggregation_method'] == method]
            if len(method_data) > 0:
                ax.plot(method_data['round'], method_data['best_recall_so_far'], 
                       marker='s', label=method, linewidth=2, markersize=4)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Mejor Recall', fontsize=12)
        ax.set_title('Mejor Recall Hasta la Ronda', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Delta Recall (Mejora)
        ax = axes[1, 0]
        for method in methods:
            method_data = self.server_data[self.server_data['aggregation_method'] == method]
            if len(method_data) > 0:
                ax.plot(method_data['round'], method_data['delta_recall'], 
                       marker='^', label=method, linewidth=2, markersize=4, alpha=0.7)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Sin mejora')
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Delta Recall', fontsize=12)
        ax.set_title('Mejora de Recall por Ronda', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Patience Counter
        ax = axes[1, 1]
        for method in methods:
            method_data = self.server_data[self.server_data['aggregation_method'] == method]
            if len(method_data) > 0:
                ax.plot(method_data['round'], method_data['patience_counter'], 
                       marker='d', label=method, linewidth=2, markersize=4)
        
        if len(self.server_data) > 0:
            patience_limit = self.server_data['patience_value'].iloc[0]
            ax.axhline(y=patience_limit, color='red', linestyle='--', 
                      alpha=0.5, label=f'Límite (patience={patience_limit})')
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Contador de Paciencia', fontsize=12)
        ax.set_title('Rondas Sin Mejora (Early Stopping)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {save_path}")
        plt.close()
    
    def plot_worker_metrics(self, save_path='worker_metrics.png'):
        """Gráficas de métricas de workers"""
        if not self.worker_data:
            print("⚠️ No hay datos de workers")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Métricas de Workers Locales', fontsize=16, fontweight='bold')
        
        # Combinar datos de todos los workers
        all_workers = pd.concat(self.worker_data.values(), ignore_index=True)
        
        # 1. Accuracy Local
        ax = axes[0, 0]
        for agent_id, data in self.worker_data.items():
            ax.plot(data['round'], data['local_accuracy'], 
                   marker='o', label=f'Worker {agent_id}', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Accuracy Local', fontsize=12)
        ax.set_title('Accuracy Local por Worker', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Recall Local
        ax = axes[0, 1]
        for agent_id, data in self.worker_data.items():
            ax.plot(data['round'], data['local_recall'], 
                   marker='s', label=f'Worker {agent_id}', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Recall Local', fontsize=12)
        ax.set_title('Recall Local por Worker', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. F1-Score Local
        ax = axes[1, 0]
        for agent_id, data in self.worker_data.items():
            ax.plot(data['round'], data['local_f1'], 
                   marker='^', label=f'Worker {agent_id}', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('F1-Score Local', fontsize=12)
        ax.set_title('F1-Score Local por Worker', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 4. Loss Local
        ax = axes[1, 1]
        for agent_id, data in self.worker_data.items():
            ax.plot(data['round'], data['local_loss'], 
                   marker='d', label=f'Worker {agent_id}', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Loss Local', fontsize=12)
        ax.set_title('Loss Local por Worker', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {save_path}")
        plt.close()
    
    def plot_communication_cost(self, save_path='communication_cost.png'):
        """Gráficas de costo de comunicación"""
        if not self.worker_data:
            print("⚠️ No hay datos de workers")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Costo de Comunicación', fontsize=16, fontweight='bold')
        
        # 1. Bytes Totales por Ronda (todos los workers)
        ax = axes[0, 0]
        for agent_id, data in self.worker_data.items():
            ax.plot(data['round'], data['local_total_bytes'] / 1024,  # KB
                   marker='o', label=f'Worker {agent_id}', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Bytes Totales (KB)', fontsize=12)
        ax.set_title('Bytes Comunicados por Ronda', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Upload vs Download
        ax = axes[0, 1]
        all_workers = pd.concat(self.worker_data.values(), ignore_index=True)
        upload_by_round = all_workers.groupby('round')['local_upload_bytes'].sum() / 1024
        download_by_round = all_workers.groupby('round')['local_download_bytes'].sum() / 1024
        
        rounds = upload_by_round.index
        ax.bar(rounds - 0.2, upload_by_round.values, width=0.4, label='Upload', alpha=0.7)
        ax.bar(rounds + 0.2, download_by_round.values, width=0.4, label='Download', alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Bytes (KB)', fontsize=12)
        ax.set_title('Upload vs Download por Ronda', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Bytes Acumulados
        ax = axes[1, 0]
        for agent_id, data in self.worker_data.items():
            cumulative_bytes = data['local_total_bytes'].cumsum() / (1024 * 1024)  # MB
            ax.plot(data['round'], cumulative_bytes, 
                   marker='s', label=f'Worker {agent_id}', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Bytes Acumulados (MB)', fontsize=12)
        ax.set_title('Bytes Totales Acumulados', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 4. Tiempo de Comunicación
        ax = axes[1, 1]
        all_workers = pd.concat(self.worker_data.values(), ignore_index=True)
        comm_time_by_round = all_workers.groupby('round').apply(
            lambda x: (pd.to_numeric(x['local_send_time'], errors='coerce') + 
                      pd.to_numeric(x['local_receive_time'], errors='coerce')).mean()
        )
        
        ax.plot(comm_time_by_round.index, comm_time_by_round.values, 
               marker='o', linewidth=2, markersize=4, color='purple')
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Tiempo (segundos)', fontsize=12)
        ax.set_title('Tiempo Promedio de Comunicación por Ronda', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {save_path}")
        plt.close()
    
    def plot_training_time(self, save_path='training_time.png'):
        """Gráficas de tiempo de entrenamiento"""
        if not self.worker_data:
            print("⚠️ No hay datos de workers")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Tiempo de Entrenamiento', fontsize=16, fontweight='bold')
        
        # 1. Tiempo de entrenamiento por worker
        ax = axes[0]
        for agent_id, data in self.worker_data.items():
            train_times = pd.to_numeric(data['local_train_time'], errors='coerce')
            ax.plot(data['round'], train_times, 
                   marker='o', label=f'Worker {agent_id}', linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Tiempo (segundos)', fontsize=12)
        ax.set_title('Tiempo de Entrenamiento Local por Ronda', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Estadísticas de tiempo
        ax = axes[1]
        all_workers = pd.concat(self.worker_data.values(), ignore_index=True)
        train_times = pd.to_numeric(all_workers['local_train_time'], errors='coerce')
        
        time_stats_by_round = all_workers.groupby('round')['local_train_time'].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        ).groupby(level=0).agg(['mean', 'std'])
        
        rounds = time_stats_by_round.index
        means = time_stats_by_round['mean']
        stds = time_stats_by_round['std']
        
        ax.plot(rounds, means, marker='o', linewidth=2, markersize=4, label='Media')
        ax.fill_between(rounds, means - stds, means + stds, alpha=0.3, label='±1 std')
        
        ax.set_xlabel('Ronda', fontsize=12)
        ax.set_ylabel('Tiempo (segundos)', fontsize=12)
        ax.set_title('Tiempo Promedio de Entrenamiento (con desviación)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {save_path}")
        plt.close()
    
    def generate_summary_report(self, save_path='summary_report.txt'):
        """Genera reporte resumen con estadísticas"""
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE ANÁLISIS DE APRENDIZAJE FEDERADO\n")
            f.write("="*80 + "\n\n")
            
            if self.server_data is not None:
                f.write("--- MÉTRICAS DEL SERVIDOR ---\n")
                for method in self.server_data['aggregation_method'].unique():
                    method_data = self.server_data[self.server_data['aggregation_method'] == method]
                    f.write(f"\nMétodo: {method}\n")
                    f.write(f"  Total de rondas: {len(method_data)}\n")
                    f.write(f"  Mejor recall: {method_data['best_recall_so_far'].max():.4f}\n")
                    f.write(f"  Recall final: {method_data['global_recall'].iloc[-1]:.4f}\n")
                    f.write(f"  Rondas sin mejora al final: {method_data['patience_counter'].iloc[-1]}\n")
                    
                    if method_data['early_stopping_triggered'].iloc[-1] == 1:
                        f.write(f"  Early stopping: SÍ\n")
                        f.write(f"  Razón: {method_data['early_stopping_reason'].iloc[-1]}\n")
                    else:
                        f.write(f"  Early stopping: NO\n")
                
                f.write("\n")
            
            if self.worker_data:
                f.write("--- MÉTRICAS DE WORKERS ---\n")
                all_workers = pd.concat(self.worker_data.values(), ignore_index=True)
                
                f.write(f"\nNúmero de workers: {len(self.worker_data)}\n")
                f.write(f"Total de rondas registradas: {len(all_workers)}\n\n")
                
                f.write("Métricas finales promedio:\n")
                f.write(f"  Accuracy: {all_workers['local_accuracy'].mean():.4f} ± {all_workers['local_accuracy'].std():.4f}\n")
                f.write(f"  Precision: {all_workers['local_precision'].mean():.4f} ± {all_workers['local_precision'].std():.4f}\n")
                f.write(f"  Recall: {all_workers['local_recall'].mean():.4f} ± {all_workers['local_recall'].std():.4f}\n")
                f.write(f"  F1-Score: {all_workers['local_f1'].mean():.4f} ± {all_workers['local_f1'].std():.4f}\n")
                f.write(f"  Loss: {all_workers['local_loss'].mean():.4f} ± {all_workers['local_loss'].std():.4f}\n")
                
                f.write("\nCosto de comunicación:\n")
                total_bytes = all_workers['local_total_bytes'].sum()
                f.write(f"  Bytes totales: {total_bytes / (1024*1024):.2f} MB\n")
                f.write(f"  Bytes por ronda (promedio): {all_workers['local_total_bytes'].mean() / 1024:.2f} KB\n")
                
                f.write("\nTiempos:\n")
                train_times = pd.to_numeric(all_workers['local_train_time'], errors='coerce')
                f.write(f"  Tiempo de entrenamiento (promedio): {train_times.mean():.2f} ± {train_times.std():.2f} s\n")
                
                f.write("\n")
        
        print(f"✓ Reporte guardado: {save_path}")
    
    def run_full_analysis(self, output_dir='analysis_results'):
        """Ejecuta análisis completo y guarda todas las gráficas"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("INICIANDO ANÁLISIS COMPLETO")
        print("="*80 + "\n")
        
        self.plot_convergence_comparison(save_path=os.path.join(output_dir, 'convergence_comparison.png'))
        self.plot_worker_metrics(save_path=os.path.join(output_dir, 'worker_metrics.png'))
        self.plot_communication_cost(save_path=os.path.join(output_dir, 'communication_cost.png'))
        self.plot_training_time(save_path=os.path.join(output_dir, 'training_time.png'))
        self.generate_summary_report(save_path=os.path.join(output_dir, 'summary_report.txt'))
        
        print("\n" + "="*80)
        print(f"✓ ANÁLISIS COMPLETADO - Resultados en: {output_dir}/")
        print("="*80 + "\n")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python analyze_results.py <directorio_de_resultados> [prefijo]")
        print("Ejemplo: python analyze_results.py . ")
        print("Ejemplo con prefijo: python analyze_results.py . experiment1_")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else ''
    
    analyzer = FederatedAnalyzer(results_dir)
    analyzer.load_data(prefix)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
