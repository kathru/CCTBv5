# Relatório de Simulação CCTBv5 — Jan-Abr 2026

## 1. Sumário Executivo
Este relatório apresenta os resultados da simulação do robô de trading **CCTBv5** utilizando a estratégia `MomentumStrategy` no período de **1° de janeiro de 2026 a 30 de abril de 2026** (120 dias).

| Métrica | Valor |
| :--- | :--- |
| **Período** | 01/01/2026 a 30/04/2026 |
| **Capital Inicial** | $ 96.592,87 |
| **Capital Final** | $ 96.172,25 |
| **P&L Total** | -$ 420,62 |
| **Retorno Total** | -0,44% |
| **Total de Trades** | 171 |
| **Win Rate Global** | 28,7% |
| **Total de Taxas** | $ 644,31 |

---

## 2. Resultados por Ativo
O capital inicial foi dividido igualmente entre os três ativos ($ 32.197,62 cada).

### BTC-USDT
*   **P&L:** +$ 13,81 (+0,04%)
*   **Win Rate:** 30,4%
*   **Profit Factor:** 1,03
*   **Trades:** 56
*   **Capital Final:** $ 32.211,43

### ETH-USDT
*   **P&L:** +$ 194,27 (+0,60%)
*   **Win Rate:** 32,0%
*   **Profit Factor:** 1,44
*   **Trades:** 50
*   **Capital Final:** $ 32.391,89

### SOL-USDT
*   **P&L:** -$ 628,69 (-1,95%)
*   **Win Rate:** 24,6%
*   **Profit Factor:** 0.33
*   **Trades:** 65
*   **Capital Final:** $ 31.568,93

---

## 3. Análise de Performance

1.  **Impacto das Taxas:** O P&L total foi negativo em $ 420,62, porém as taxas pagas somaram $ 644,31. Isso indica que, em termos de "Gross P&L", a estratégia foi positiva, mas a frequência de trades e as margens capturadas não foram suficientes para superar o custo operacional (fricção).
2.  **Divergência de Ativos:** Enquanto BTC e ETH conseguiram manter a neutralidade ou lucro leve, o ativo **SOL-USDT** apresentou uma performance substancialmente inferior, com um Profit Factor de apenas 0,33 e um prejuízo que anulou os ganhos dos outros dois ativos.
3.  **Mecanismos de Proteção:** Durante a simulação, observou-se um alto volume de bloqueios pelo `liquidity_gate` (EdgeConditioner). Isso mostra que o robô foi eficaz em evitar mercados com baixa liquidez ou "secos", o que preveniu perdas maiores por slippage em períodos desfavoráveis.
4.  **Perfil da Estratégia:** O Win Rate baixo (sub-30%) é característico de estratégias de momentum que buscam capturar grandes movimentos (convexidade). Entretanto, no período testado, os "vencedores" não foram grandes o suficiente para compensar a série de pequenos prejuízos e as taxas.

---

## 4. Recomendações

1.  **Otimização do Sizing para SOL:** Investigar o comportamento específico de SOL-USDT no período. A volatilidade do ativo pode exigir thresholds de entrada mais conservadores ou um position sizing reduzido em comparação ao BTC/ETH.
2.  **Redução de Ruído:** Elevar levemente o threshold de `score` mínimo para entrada. Isso reduziria o número total de trades (atualmente ~1.4 trades/dia por ativo), diminuindo o impacto cumulativo das taxas e focando apenas em sinais de altíssima convicção.
3.  **Revisão do Take Profit em Regimes de CHOP:** Como grande parte das perdas ocorreu em mercados laterais onde as taxas pesam mais, considerar saídas mais rápidas ou evitar completamente entradas quando o regime detectado for `MEAN_REVERTING_CHOP` com baixa volatilidade.
4.  **Ajuste de Taxas:** A simulação utilizou uma taxa de 0,15% (Taker). Caso o usuário possua um tier de taxas menor na OKX (ex: VIP ou uso de rebate), a estratégia passaria a ser lucrativa no consolidado. Recomenda-se verificar o tier real da conta.

---
*Relatório gerado em simulação local determinística.*
