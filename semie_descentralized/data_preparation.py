"""
Data Preparation for Tabular NCD Federated Learning.

Este módulo prepara los datos tabulares de defunciones para entrenamiento federado.
Cada nodo (hospital) tiene su propio CSV con datos locales.

Pasos:
 1. Cargar datos crudos del hospital (dataX.csv)
 2. Cargar preprocessor global compartido (preprocessor_global.joblib)
 3. Transformar features a matriz numérica
 4. Dividir en train/val/test con estratificación
 5. Guardar CSVs procesados
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import logging


def load_preprocessor(preprocessor_path):
    """Carga el preprocessor desde archivo joblib"""
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor no encontrado: {preprocessor_path}")
    return joblib.load(preprocessor_path)


def get_default_config(base_dir: str, agent_name: str = "a1") -> dict:
    """
    Genera configuración por defecto para el preprocesamiento.
    
    Estructura esperada en deploy_node (cada Raspberry Pi):
      deploy_node/
        data/
          data.csv          <- UN SOLO archivo por nodo
        artifacts/
          preprocessor_global.joblib
        fl_main/examples/tabular_ncd/  <- este módulo
    """
    # Subir desde fl_main/examples/tabular_ncd/ hasta deploy_node/
    deploy_root = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
    
    # Cada nodo tiene UN SOLO archivo: data.csv
    raw_data_path = os.path.join(deploy_root, "data", "data.csv")
    
    # Preprocessor compartido
    preprocessor_path = os.path.join(deploy_root, "artifacts", "preprocessor_global.joblib")
    
    # Directorio de salida para CSVs procesados
    output_dir = os.path.join(deploy_root, "data", "processed")
    
    return {
        "raw_data_path": raw_data_path,
        "preprocessor_path": preprocessor_path,
        "target_col": "is_premature_ncd",
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "random_state": 42,
        "output_dir": output_dir,
        "rename_target_to": "target",
        "drop_cols": ["hospital_cliente"],
        "balance_strategy": "none"
    }


def simple_undersample(X: np.ndarray, y: np.ndarray, random_state: int):
    """Undersampling simple de la clase mayoritaria."""
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    if len(pos_idx) == len(neg_idx):
        return X, y
    
    if len(pos_idx) < len(neg_idx):
        keep_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        sel = np.concatenate([pos_idx, keep_neg])
    else:
        keep_pos = rng.choice(pos_idx, size=len(neg_idx), replace=False)
        sel = np.concatenate([keep_pos, neg_idx])
    
    sel = rng.permutation(sel)
    return X[sel], y[sel]


def run_preprocessing(cfg: dict) -> dict:
    """
    Ejecuta el preprocesamiento con la configuración dada.
    Retorna un diccionario con metadata del proceso.
    """
    raw_path = cfg['raw_data_path']
    preproc_path = cfg['preprocessor_path']
    target_col = cfg['target_col']
    out_dir = cfg.get('output_dir', './data')
    os.makedirs(out_dir, exist_ok=True)
    
    rename_target = cfg.get('rename_target_to', 'target')
    drop_cols = cfg.get('drop_cols', [])
    balance_strategy = cfg.get('balance_strategy', 'none')

    train_frac = float(cfg['train_frac'])
    val_frac = float(cfg['val_frac'])
    test_frac = float(cfg['test_frac'])
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, 'Fractions must sum to 1.'
    rnd = int(cfg.get('random_state', 42))

    # 1) Cargar datos crudos
    logging.info(f"Cargando datos desde: {raw_path}")
    df = pd.read_csv(raw_path)
    
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found. Columns: {list(df.columns)}')

    # Eliminar columnas no deseadas
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # Extraer target
    y = df[target_col].astype(float).to_numpy()
    X_raw = df.drop(columns=[target_col])

    # 2) Cargar preprocessor global
    logging.info(f"Cargando preprocessor desde: {preproc_path}")
    preproc = joblib.load(preproc_path)

    # 3) Transformar features
    X_mat = preproc.transform(X_raw)
    n_features = X_mat.shape[1]
    feature_cols = [f'f{i}' for i in range(n_features)]
    logging.info(f"Features transformadas: {n_features} columnas")

    # 4) Balanceo opcional
    if balance_strategy == 'undersample_majority':
        X_mat, y = simple_undersample(X_mat, y, rnd)
        logging.info(f"Undersampling aplicado. Nuevos tamaños: {len(y)}")

    # 5) Splits estratificados
    temp_frac = val_frac + test_frac
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_mat, y, test_size=temp_frac, random_state=rnd,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    if temp_frac > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_frac / temp_frac), random_state=rnd,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
    else:
        X_val, X_test = np.empty((0, n_features)), np.empty((0, n_features))
        y_val, y_test = np.empty((0,)), np.empty((0,))

    def build_df(Xp: np.ndarray, yp: np.ndarray) -> pd.DataFrame:
        d = pd.DataFrame(Xp, columns=feature_cols)
        d[rename_target] = yp.astype(int)
        return d

    # 6) Guardar CSVs
    train_df = build_df(X_train, y_train)
    val_df = build_df(X_val, y_val)
    test_df = build_df(X_test, y_test)

    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    logging.info(f"CSVs guardados en: {out_dir}")
    logging.info(f"  - train.csv: {len(train_df)} samples")
    logging.info(f"  - val.csv: {len(val_df)} samples")
    logging.info(f"  - test.csv: {len(test_df)} samples")

    return {
        'n_features_transformed': n_features,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'output_dir': out_dir
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test con configuración por defecto
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = get_default_config(base_dir, agent_name="a1")
    
    print("Configuración generada:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    
    if os.path.exists(cfg['raw_data_path']) and os.path.exists(cfg['preprocessor_path']):
        meta = run_preprocessing(cfg)
        print("\nMetadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
    else:
        print("\n⚠️ Archivos no encontrados. Verifica las rutas.")
