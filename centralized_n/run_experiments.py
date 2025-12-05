"""
Script auxiliar para ejecutar experimentos batch de FedAvg vs FedProx
Ejecuta múltiples configuraciones y organiza resultados
"""
import os
import subprocess
import time
import shutil
from datetime import datetime


def run_experiment(experiment_name, method, approach='centralized'):
    """Ejecuta un experimento completo"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENTO: {experiment_name}")
    print(f"Método: {method}, Enfoque: {approach}")
    print(f"{'='*80}\n")
    
    # Directorio de trabajo
    work_dir = f"{approach}/"
    
    # Crear directorio de resultados
    results_dir = f"experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Backup config actual
    shutil.copy(f"{work_dir}config.py", f"{results_dir}/config_backup.py")
    
    # Modificar config.py temporalmente
    with open(f"{work_dir}config.py", 'r') as f:
        config_content = f.read()
    
    # Cambiar método de agregación
    config_content = config_content.replace(
        "AGGREGATION_METHOD = 'FedAvg'",
        f"AGGREGATION_METHOD = '{method}'"
    ).replace(
        "AGGREGATION_METHOD = 'FedProx'",
        f"AGGREGATION_METHOD = '{method}'"
    )
    
    with open(f"{work_dir}config.py", 'w') as f:
        f.write(config_content)
    
    print(f"✓ Config modificado: AGGREGATION_METHOD = '{method}'")
    print(f"✓ Directorio de resultados: {results_dir}")
    print("\nInicia el experimento manualmente:")
    
    if approach == 'centralized':
        print(f"\n1. cd {work_dir}")
        print("2. python server_centralized.py")
        print("3. En otras terminales:")
        print("   python worker_centralized.py agent_1 data1.csv")
        print("   python worker_centralized.py agent_2 data2.csv")
        print("   python worker_centralized.py agent_3 data3.csv")
        print("   python worker_centralized.py agent_4 data4.csv")
    else:
        print(f"\n1. cd {work_dir}")
        print("2. python server.py")
        print("3. En otras terminales:")
        print("   python agent_csv.py agent_1 data1.csv")
        print("   python agent_csv.py agent_2 data2.csv")
        print("   python agent_csv.py agent_3 data3.csv")
        print("   python agent_csv.py agent_4 data4.csv")
    
    print("\n4. Cuando termine, presiona ENTER para mover los CSVs...")
    input()
    
    # Mover CSVs generados
    csv_files = [
        f for f in os.listdir(work_dir)
        if f.endswith('_log.csv') or f.endswith('_log_centralized.csv')
    ]
    
    for csv_file in csv_files:
        src = os.path.join(work_dir, csv_file)
        dst = os.path.join(results_dir, csv_file)
        shutil.move(src, dst)
        print(f"✓ Movido: {csv_file} -> {results_dir}/")
    
    # Mover modelos guardados si existen
    models_dir = os.path.join(work_dir, 'models')
    if os.path.exists(models_dir):
        dst_models = os.path.join(results_dir, 'models')
        shutil.copytree(models_dir, dst_models, dirs_exist_ok=True)
        print(f"✓ Modelos copiados a {results_dir}/models/")
    
    print(f"\n✓ Experimento completado: {experiment_name}")
    print(f"✓ Resultados guardados en: {results_dir}")
    
    return results_dir


def analyze_experiment(results_dir):
    """Analiza resultados de un experimento"""
    print(f"\n{'='*80}")
    print(f"ANALIZANDO: {results_dir}")
    print(f"{'='*80}\n")
    
    output_dir = os.path.join(results_dir, 'analysis')
    
    cmd = f"python analyze_results.py {results_dir} > {os.path.join(output_dir, 'analysis.log')} 2>&1"
    
    print(f"Ejecutando: {cmd}")
    subprocess.run(cmd, shell=True)
    
    print(f"✓ Análisis completado: {output_dir}/")


def create_experiment_plan():
    """Crea un plan de experimentos"""
    plan = [
        {'name': 'central_fedavg', 'method': 'FedAvg', 'approach': 'centralized'},
        {'name': 'central_fedprox', 'method': 'FedProx', 'approach': 'centralized'},
        {'name': 'semidesc_fedavg', 'method': 'FedAvg', 'approach': 'new_semidescentralized'},
        {'name': 'semidesc_fedprox', 'method': 'FedProx', 'approach': 'new_semidescentralized'},
    ]
    
    print("\n" + "="*80)
    print("PLAN DE EXPERIMENTOS")
    print("="*80)
    
    for i, exp in enumerate(plan, 1):
        print(f"{i}. {exp['name']}: {exp['method']} en {exp['approach']}")
    
    print("\nEste script te guiará para ejecutar cada experimento.")
    print("Tendrás que iniciar manualmente el servidor y los workers/agents.")
    print("\nPresiona ENTER para continuar...")
    input()
    
    return plan


def main():
    print("\n" + "="*80)
    print("EJECUTOR DE EXPERIMENTOS - FedAvg vs FedProx")
    print("="*80)
    
    # Crear plan
    plan = create_experiment_plan()
    
    # Crear directorio base de experimentos
    os.makedirs('experiments', exist_ok=True)
    
    # Ejecutar cada experimento
    results_dirs = []
    for exp in plan:
        results_dir = run_experiment(exp['name'], exp['method'], exp['approach'])
        results_dirs.append(results_dir)
        
        print("\n¿Continuar con siguiente experimento? (s/n): ", end='')
        if input().lower() != 's':
            break
    
    # Analizar resultados
    print("\n" + "="*80)
    print("ANÁLISIS DE RESULTADOS")
    print("="*80)
    print("\n¿Analizar todos los experimentos ahora? (s/n): ", end='')
    
    if input().lower() == 's':
        for results_dir in results_dirs:
            analyze_experiment(results_dir)
    
    # Resumen final
    print("\n" + "="*80)
    print("EXPERIMENTOS COMPLETADOS")
    print("="*80)
    print("\nResultados guardados en:")
    for results_dir in results_dirs:
        print(f"  - {results_dir}")
    
    print("\nPara comparar métodos, revisa:")
    print("  - analysis_results/convergence_comparison.png")
    print("  - analysis_results/summary_report.txt")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Experimentos interrumpidos por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
