# Titanic — Machine Learning from Disaster

Baseline com Gradient Boosting (CV accuracy: 0.836).

## Como reproduzir
1. Python 3.10 + venv
2. `pip install -r requirements.txt`
3. Baixar `train.csv` e `test.csv` do Kaggle e colocar em `data/`
4. Rodar `notebooks/01_eda.ipynb`

## Estrutura
- `notebooks/01_eda.ipynb` — EDA + baseline
- `outputs/submission_baseline.csv` — primeira submissão
- `requirements.txt` — dependências

## 📊 Resumo Final – Competição Titanic Kaggle

**Objetivo:** Melhorar o *recall* e o equilíbrio geral das métricas, mantendo consistência no pipeline e explorando diferentes estratégias.

| # | Estratégia | Métricas CV (principais) | Threshold | Pontuação Kaggle |
|---|------------|--------------------------|-----------|------------------|
| 1 | Baseline (GradBoost melhor modelo inicial) | F1=0.77 aprox., Recall=0.79 | default | **0.75358** |
| 2 | Otimização de F2 (LogReg) | F2=0.803, Recall=0.836 | 0.397 | 0.72009 |
| 3 | Ajuste de threshold p/ Recall≈0.80 | F1≈0.774, Recall=0.801 | 0.490 | 0.75119 |
| 4 | Otimização incremental | F1≈0.766, Recall=0.781 | ajustado | 0.76315 |
| 5 | Ensemble Voting (soft, pesos ajustados) | F1=0.792, Recall=0.789 | 0.460 | 0.74880 |
| 6 | Ajuste de pesos iguais no voting | F1=0.790, Recall=0.795 | 0.450 | 0.74641 |
| 7 | Ajuste p/ maior precisão | F1=0.782, Recall=0.766 | 0.450 | 0.76315 |
| 8 | XGBoost otimizado | F1=0.795, Recall=0.804 | 0.490 | 0.75598 |
| 9 | LightGBM otimizado | F1≈0.79 (CV) | ajustado | 0.75358 |
| 10 | Blend LogReg + GradBoost + LightGBM (1-1-2) | F1=0.798, Recall≈0.79 | 0.54 | **0.75837** |

---

**📈 Observações finais:**
- A **maior pontuação** foi **0.76315** com abordagem de ajuste incremental.
- O *blend final* superou a média das submissões intermediárias, mostrando potencial de ensembles mais refinados.
- O uso de **tuning de threshold** foi decisivo para equilibrar recall e precisão.
- LightGBM e XGBoost se destacaram como modelos mais estáveis.
