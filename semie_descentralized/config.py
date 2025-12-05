"""
Archivo de Configuración para Aprendizaje Federado Semi-Descentralizado

IMPORTANTE: Ajustar estos valores según el entorno:
- Computadoras locales rápidas: Usar valores por defecto
- Raspberry Pi / Redes lentas: Aumentar tiempos y reintentos
"""


class Config:
    """Configuración centralizada del sistema"""
    
    # ==================== SERVIDOR ====================
    SERVER_HOST = '0.0.0.0'         # IP del servidor (0.0.0.0 para escuchar en todas las interfaces)
    SERVER_PORT = 8765              # Puerto del servidor central
    MIN_AGENTS = 4                  # Mínimo de agentes para comenzar
    STARTUP_DELAY = 2               # Segundos de espera al iniciar primera ronda
    ROUND_START_DELAY = 1           # Segundos de espera entre rondas
    
    # ==================== AGENTES ====================
    # Conexión al agregador
    MAX_RETRIES = 15                # Número máximo de intentos para conectar al agregador
    RETRY_DELAY = 3                 # Segundos entre reintentos de conexión
    
    # Entrenamiento
    LOCAL_EPOCHS = 3                # Épocas de entrenamiento local por ronda
    INTERNAL_ROUNDS = 5             # Rondas internas antes de re-seleccionar agregador
    BATCH_SIZE = 16                 # Tamaño de batch para entrenamiento
    LEARNING_RATE = 0.01            # Tasa de aprendizaje
    
    # Dataset
    DATASET_SIZE = 100              # Número de muestras por agente
    
    # WebSocket
    WS_TIMEOUT = 60                 # Timeout de conexión WebSocket en segundos
    WS_MAX_SIZE = 10 * 1024 * 1024  # Tamaño máximo de mensaje (10 MB)
    
    # Puertos de agregadores (base + número de agente)
    AGGREGATOR_PORT_BASE = 9000
    
    # ==================== MODELO ====================
    INPUT_SIZE = 10                 # Tamaño de entrada del modelo
    HIDDEN_SIZE = 20                # Tamaño de capa oculta
    OUTPUT_SIZE = 2                 # Número de clases
    
    # ==================== EARLY STOPPING ====================
    EARLY_STOPPING_PATIENCE = 20    # Rondas sin mejora antes de detener
    MAX_ROUNDS = 100                # Máximo de rondas totales
    MIN_RECALL_IMPROVEMENT = 0.001  # Mejora mínima en recall para considerar progreso
    
    # ==================== AGREGACIÓN ====================
    AGGREGATION_METHOD = 'FedProx'   # Método de agregación: 'FedAvg' o 'FedProx'
    FEDPROX_MU = 0.01               # Parámetro mu para FedProx (término proximal)


class RaspberryPiConfig(Config):
    """
    Configuración optimizada para Raspberry Pi o redes lentas
    
    Para usar esta configuración:
    1. En agent.py: from config import RaspberryPiConfig as Config
    2. En server.py: from config import RaspberryPiConfig as Config
    """
    
    # Aumentar tiempos de espera
    MAX_RETRIES = 25
    RETRY_DELAY = 5
    STARTUP_DELAY = 5
    ROUND_START_DELAY = 3
    WS_TIMEOUT = 120
    
    # Reducir carga de entrenamiento
    LOCAL_EPOCHS = 2
    INTERNAL_ROUNDS = 5
    BATCH_SIZE = 8
    DATASET_SIZE = 50


class FastConfig(Config):
    """
    Configuración para equipos rápidos en red local
    """
    
    MAX_RETRIES = 10
    RETRY_DELAY = 1
    STARTUP_DELAY = 1
    ROUND_START_DELAY = 0.5
    WS_TIMEOUT = 30
    
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32


# ==================== INSTRUCCIONES DE USO ====================
"""
Para cambiar la configuración, edita la importación en agent.py y server.py:

OPCIÓN 1 - Configuración por defecto (local rápido):
    from config import Config

OPCIÓN 2 - Raspberry Pi / Red lenta:
    from config import RaspberryPiConfig as Config

OPCIÓN 3 - Equipos muy rápidos:
    from config import FastConfig as Config

OPCIÓN 4 - Personalizada:
    1. Crea tu propia clase que herede de Config
    2. Sobrescribe solo los valores que necesites
    
    class MiConfig(Config):
        MAX_RETRIES = 30
        LOCAL_EPOCHS = 10
"""
