from __future__ import annotations

import logging
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ARQUIVO_MODELO = "artifacts/modelo.pkl"


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador customizado usado no treino para criar atributos derivados.
    É mantido aqui para garantir compatibilidade do pickle em ambiente limpo.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AmesFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        def col(nome: str) -> pd.Series:
            if nome in df.columns:
                return df[nome]
            return pd.Series(0, index=df.index)

        df["TotalBath"] = (
            col("FullBath").fillna(0)
            + 0.5 * col("HalfBath").fillna(0)
            + col("BsmtFullBath").fillna(0)
            + 0.5 * col("BsmtHalfBath").fillna(0)
        )

        df["TotalSF"] = (
            col("TotalBsmtSF").fillna(0)
            + col("1stFlrSF").fillna(0)
            + col("2ndFlrSF").fillna(0)
        )

        df["IdadeImovel"] = col("YrSold").fillna(0) - col("YearBuilt").fillna(0)
        df["IdadeReforma"] = col("YrSold").fillna(0) - col("YearRemodAdd").fillna(0)

        return df


def _carregar_modelo(caminho_modelo: str):
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Artefato '{caminho_modelo}' não encontrado.")
    logger.info("Carregando artefato serializado: %s", caminho_modelo)
    return joblib.load(caminho_modelo)


def _aplicar_feature_engineering_se_necessario(modelo, df: pd.DataFrame) -> pd.DataFrame:
    """
    Evita dupla aplicação de feature engineering.
    Se o modelo já tiver a etapa 'feature_engineering' embutida, envia o DF cru.
    Caso contrário, aplica o transformador antes da inferência.
    """
    alvo = modelo

    if hasattr(modelo, "regressor") and modelo.regressor is not None:
        alvo = modelo.regressor

    if hasattr(alvo, "named_steps") and "feature_engineering" in alvo.named_steps:
        logger.info("Modelo já contém feature engineering interna. Usando base bruta.")
        return df

    logger.info("Modelo não contém feature engineering interna. Aplicando transformações externas.")
    return AmesFeatureEngineer().transform(df)


def prever_precos(caminho_arquivo_teste: str, caminho_modelo: str = ARQUIVO_MODELO) -> np.ndarray:
    logger.info("Iniciando inferência para o arquivo: %s", caminho_arquivo_teste)

    if not os.path.exists(caminho_arquivo_teste):
        raise FileNotFoundError(f"Arquivo de teste não encontrado: {caminho_arquivo_teste}")

    df_teste = pd.read_csv(caminho_arquivo_teste)
    logger.info("Base carregada com %d linhas e %d colunas.", df_teste.shape[0], df_teste.shape[1])

    X_teste = df_teste.drop(columns=["Id"], errors="ignore")

    modelo = _carregar_modelo(caminho_modelo)
    X_preparado = _aplicar_feature_engineering_se_necessario(modelo, X_teste)

    logger.info("Executando predições...")
    predicoes = modelo.predict(X_preparado)

    saida = np.asarray(predicoes, dtype=float).reshape(-1)
    logger.info("Inferência concluída com sucesso. Total de predições: %d", saida.shape[0])
    return saida


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" INICIANDO TESTE LOCAL DA ESTEIRA DE INFERÊNCIA")
    print("=" * 65 + "\n")

    caminho_teste = "data/raw/teste.csv" # Garanta que aponta para o dataset correto

    try:
        print("[Etapa 1/2] Carregando modelo e aplicando pré-processamento...")
        previsoes = prever_precos(caminho_teste)
        
        print("\n[Etapa 2/2] Processamento concluído!")
        print("\n" + "=" * 65)
        print("RESUMO DAS PREDIÇÕES ADQUIRIDAS ")
        print("=" * 65)
        
        # Transformando as predições em um DataFrame para facilitar a extração de métricas
        # Supondo que a contagem do ID dos imóveis de teste comece a partir de 1461 (padrão Ames Housing)
        df_resultados = pd.DataFrame({"Preco_Venda_Estimado": previsoes})
        
        # Exibição de estatísticas descritivas para conferência rápida (Sanity Check)
        print("\nESTATÍSTICAS DESCRITIVAS (em Dólares):")
        print(f"   -> Total de Imóveis Avaliados : {len(previsoes)}")
        print(f"   -> Preço Médio Estimado     : ${np.mean(previsoes):,.2f}")
        print(f"   -> Preço Mínimo Estimado    : ${np.min(previsoes):,.2f}")
        print(f"   -> Preço Máximo Estimado    : ${np.max(previsoes):,.2f}")
        
        print("\nAMOSTRA DAS 5 PRIMEIRAS PREDIÇÕES:")
        print("-" * 40)
        # Formata a coluna para moeda antes de imprimir
        df_amostra = df_resultados.head().copy()
        df_amostra['Preco_Venda_Estimado'] = df_amostra['Preco_Venda_Estimado'].apply(lambda x: f"${x:,.2f}")
        print(df_amostra.to_string(index=False))
        print("-" * 40 + "\n")

        caminho_submissao = "artifacts/submissao_final.csv"
        df_resultados.to_csv(caminho_submissao, index=False)
        print(f"Arquivo de submissão salvo automaticamente em: {caminho_submissao}\n")

    except Exception as e:
        logger.error("Erro crítico durante a execução do pipeline.")
        logger.error(str(e))
        sys.exit(1)
