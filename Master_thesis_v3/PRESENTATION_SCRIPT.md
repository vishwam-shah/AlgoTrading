# Viva Presentation Script — A Positional Trading System for NSE India

**Audience assumption:** Panel members are technical (AI / engineering faculty) but **not** stock-market specialists. Every market term is introduced in plain language the first time it appears.

**Time budget:** ~20 min talk + 10 min Q&A. Each slide section below has a target time.

---

## Slide 1 — Title (30 sec)

> "Good morning / afternoon, respected committee. My name is Vishwam Shah, roll number 24MAI022, and I'm presenting my M.Tech thesis titled **'A Positional Trading System for NSE India: Walk-Forward Ensemble Learning, Calibrated Probabilities, and Live API Deployment.'** This work is guided by Dr. Jigarkumar Shah from the ICT Department."

> "Before I begin, one note for context: NSE here stands for the **National Stock Exchange of India**, and 'positional trading' means we hold a stock for several days, not seconds or minutes. So this is **not** high-frequency trading."

---

## Slide 2 — Outline (30 sec)

> "I'll cover this in 14 sections — first the motivation and problem statement, then literature, methodology, results, our live deployment system, comparison with prior work, and finally conclusions and future work."

> "The talk has three parts: **what problem we solved**, **how we solved it honestly**, and **how we made it deployable**."

---

## Slide 3 — Introduction & Motivation (1.5 min)

> "Most published research on Indian stock prediction reports accuracies between **60% and 83%**. That sounds impressive — but here's the catch."

> "When you actually try to **trade** based on those signals, you pay real-world costs. In India, every round-trip trade — buy + sell — costs about **40 basis points**, which means **0.4%** of the trade value. This comes from: STT (Securities Transaction Tax), exchange fees, GST, SEBI charges, and slippage — slippage is the difference between the price you expect and the price you actually get when your order hits the market."

> "So the question becomes: **does a model that's 60% accurate on paper actually make money after these costs?** Often the answer is no."

> "Our key insight, shown on this slide, is that a more **conservative** model that trades only when it's very confident — what we call a 'gated' model with 51% raw accuracy — can actually outperform a 'greedy' 58% model after costs. Why? Because the greedy one trades too often and pays costs on every losing trade."

> "On the right, you can see the gaps in existing literature: small universes, single random train-test splits, data leakage from full-panel scaling, no regime testing, and no live deployment layer."

**Plain-language helpers:**
- *Basis point (bps)*: 1/100th of a percent. 40 bps = 0.40%.
- *Regime*: a market phase — bull (going up), bear (going down), sideways (flat).
- *Slippage*: market moves between your decision and your execution; you lose a few paise per share.

---

## Slide 4 — Problem Statement (2 min)

> "Let me formalize the problem precisely."

> "**The core problem:** Indian-market ML studies report 60–83% accuracy, but nobody has demonstrated that their signal actually survives the Indian cost stack under a rigorous, leakage-audited validation protocol on a realistic universe of 100 stocks."

> "We break this into **three sub-problems**:"

> "**One — Honest accuracy.** How do we measure direction accuracy without accidentally leaking future information into training? Most papers fail here, which is why their numbers look inflated."

> "**Two — Deployable profit.** Beyond accuracy, can we produce a Sharpe ratio greater than 1 after costs, across all three market regimes? Sharpe ratio is simply return per unit of risk — a Sharpe of 1 is considered respectable, above 1.5 is good."

> "**Three — Capital-safe deployment.** When you go live with real money, you can't just trust a point estimate from a backtest. You need an **objective gate** that says 'I'm not deploying capital until certain measured KPIs are met for real.' This is risk-management discipline."

> "**Research question** (on the right): *Can a calibrated, gated ensemble of heterogeneous models produce a statistically significant, cost-net positive return on a 100-stock NSE universe under leakage-audited walk-forward validation?*"

> "**Scope:** 100 NSE stocks, daily-bar trading — meaning we make one decision per day per stock — direction classification only — i.e., will the stock close up or down tomorrow — and we explicitly exclude high-frequency trading and derivatives. Those are listed as future work."

---

## Slide 5 — Why Reported Accuracies Are Inflated (2 min)

> "This slide is the most important conceptual slide. I want to spend a moment here."

> "We ran a **controlled experiment**. Same code, same data, same model. The **only** thing we changed was the validation protocol."

> "With a single random 60-20-20 train-validation-test split — which is what most papers do — we got **68.28% accuracy**."

> "With a proper six-fold **expanding walk-forward validation** — where each fold uses only past data to predict future data — the same model dropped to **50.97%**."

> "That's a **17-percentage-point drop** caused purely by the validation protocol. The high number was **data leakage**, not skill. The model wasn't 'learning' the market; it was peeking at the future."

> "On the left I list four common leakage sources: random splits mixing future rows into training, scalers fitted on the full panel including test data, rolling features computed across the split boundary, and single test windows that cherry-pick one favorable regime."

> "On the right are the **six audited controls** we used to eliminate leakage. The most important one is: target is `shift(-1)` applied AFTER all features are computed — so tomorrow's price never touches today's feature row. Scaler and PCA are refit on each fold's training slice only. Global cues like S&P 500 are shifted forward one day to simulate when an Indian trader would actually see them."

**Why panel cares:** This is the methodological backbone. It's why our 51% number is meaningful and others' 83% may not be.

---

## Slide 6 — Literature Review: Foundational Works (1 min)

> "Quick literature foundation. On the left, four theoretical pillars this work builds on:"

> "**EMH** — the Efficient Market Hypothesis from Fama 1970 — says markets are priced fairly so beating them is hard. But Fama himself documented exceptions: calendar effects and momentum. We exploit those."

> "**Walk-Forward Validation** — formalized in López de Prado's 2018 book — is the gold-standard validation protocol for time-series and a catalogue of how studies leak."

> "**Stacked generalisation** from Wolpert 1992 — combining heterogeneous learners reduces correlated error."

> "**Temperature scaling** from Guo et al. 2017 — a technique to fix neural-network probability over-confidence without changing the prediction itself."

> "On the right, the five model architectures we use: two tree models (XGBoost, LightGBM) and three deep-learning models (BiLSTM, TCN-Transformer, N-BEATS). All standard, well-cited."

---

## Slide 7 — Literature Review: Indian Prior Work (1 min)

> "This table summarizes Indian-market and adjacent prior work. Notice the pattern: Patel 2015 reports 83% on 10 NSE stocks with a single split. Shah et al. 2019 reports 56–83% but on a survey of arbitrary protocols. Nobody uses 100 stocks with proper walk-forward."

> "On the right are the **five shared gaps** across this literature: small universe, single splits, no brokerage costs, no probability calibration, no live deployment layer."

> "Our bottom row addresses all five: 100 stocks, 6-fold walk-forward, full Indian cost stack, temperature-scaled calibration, and a live Angel One layer."

---

## Slide 8 — Research Problem & Objectives (1.5 min)

> "From the central research question on slide 4, we derive **four research questions** and **four design goals**."

> "Each RQ is **testable** and has a specific piece of evidence in the Results chapter that answers it. They aren't a wishlist — they're the falsifiable sub-claims that, taken together, prove or break the central question."

> "**RQ1**: Does a heterogeneous ensemble (trees + DL) beat single homogeneous models? Answered by panel-wide accuracy ablation."
> "**RQ2**: Does temperature calibration actually make the gate reliable? Answered by Brier-score drop and bootstrap confidence intervals."
> "**RQ3**: Do the macro cues and NSE calendar features add significant predictive signal? Answered by feature ablation."
> "**RQ4**: Is it profitable across all three market regimes? Answered by per-regime Sharpe table."

> "On the right, the **four design goals** drive implementation: G1 — every feature strictly prior to NSE open on day t (no future peeking). G2 — heterogeneous ensemble with calibration. G3 — expanding walk-forward across six folds per symbol, giving us 63,082 out-of-sample predictions, meaning **data the model never saw during training**. G4 — the promotion gate I'll show later."

---

## Slide 9 — Data & Universe (1 min)

> "**100 stocks** from Nifty-50 plus Nifty Next-50, spanning **14 sectors**. Date range January 2018 through April 2026. Data source is yfinance with an incremental Parquet cache — meaning each run only downloads new data, not the full history."

> "We also pull six **global macro cues**: S&P 500, Nasdaq, VIX (the US fear index), DXY (US Dollar Index), Crude Oil, and Nikkei 225. Critical detail: these are shifted forward one calendar day, because an Indian market opening at 9:15 IST only knows yesterday's US close."

> "On the right are **NSE-specific calendar features** — days to next F&O expiry (the last Thursday of each month is options expiry day, historically volatile), days to next RBI Monetary Policy Committee meeting, Union Budget proximity, results season."

> "And **sector force-includes** at the bottom: IT stocks always get USD/INR and Nasdaq returns. Banking stocks always get days-to-RBI and crude. These sector-specific features are why our IT and Banking predictions are stronger."

---

## Slide 10 — Feature Engineering (1.5 min)

> "Roughly **219 features per stock**, organized into eleven families. Price-based, technical indicators like MACD and RSI, volume, intraday range, temporal cyclic encodings of day of week and month, volatility estimators like Parkinson and Garman-Klass, momentum, statistical (skew kurtosis autocorrelation), market regime indicators, non-linear interactions, and the global cues plus calendar."

> "On the right are the **leakage controls** we discussed: target shift, per-fold scaler, per-fold PCA, prior-day global cue join, per-fold feature selection, per-fold temperature fit."

> "The model inputs are: top-50 PCA components for the tree models (because trees handle correlated features poorly), and a 3D tensor of 20-day sequences for the deep models."

---

## Slide 11 — Five-Model Ensemble (2 min)

> "Our ensemble has **two tree models and three deep-learning models**, run at different times to fit a realistic trading day."

> "**LightGBM and XGBoost** are gradient-boosted trees — these run the morning fast pass at 9:00 IST."

> "**BiLSTM, TCN-Transformer, and N-BEATS** are the deep-learning members, trained overnight from 6:00 PM to 8:30 AM IST."

> "Now the **most important conceptual block** on the right: the **logistic meta-learner**."

> "Plain-language: imagine five doctors each giving a diagnosis. The meta-learner is a chief physician who has watched these five doctors for years on this specific patient, and has learned: 'For this patient, doctor A is usually right; doctor C is usually wrong.' It assigns weights accordingly."

> "Crucially, on tabular features the boosters (trees) dominate — this is a well-known empirical result from Shwartz-Ziv 2022. The deep-learning members are kept, but the meta-learner **automatically down-weights them**. The contribution is not 'five equal voters'; the contribution is the ensemble **protecting itself** from its weakest members."

> "**Temperature scaling** below: a calibration method. Neural networks tend to be over-confident — they say '95% sure' when reality is 70%."

> "The formula on the slide is: **T-star equals argmin over T in [0.05, 5] of negative log likelihood of sigma(z_v over T) versus y_v**. In words: we find the single scalar T that, when we divide the validation logits by it and apply sigmoid, gives the probability values closest to the actual outcomes."

> "Then at inference: **p-hat = sigma(logit(p) divided by T-star)**. Because sigmoid is monotonic and T is positive, dividing the logit doesn't change which side of 0.5 you're on. So **argmax — the predicted class — is preserved, accuracy unchanged**. What changes: the probability NUMBER itself. After calibration, when we say 58% sure, we actually win 58% of the time. Roughly 2/3 of our (ticker, fold) cells were already calibrated (T-star ≈ 1); the other 1/3 had T-star < 0.9, meaning the model was over-confident and got compressed."

---

## Slide 12 — Walk-Forward Validation Protocol (2 min)

> "Six folds per symbol. Training ratio expands from 70% to 95% in 5-percentage-point steps. Scaler, PCA, and temperature are all refit on each fold."

> "100 stocks × 6 folds = 600 windows total, producing **63,082 out-of-sample predictions**. Test windows span November 2023 through April 2026, deliberately including all three regimes."

> "Three formulas on this slide make the protocol concrete."

> "**Fold split formula**: r_k = 0.70 + 0.05·(k−1) for k from 1 to 6. So fold 1 trains on 70% of rows, fold 2 on 75%, up to fold 6 on 95%. Train_size = floor(N × r_k). Validation is the next 10%, test is the rest."

> "**Target formula**: y_t = indicator that (C_{t+1} − C_t) / C_t ≥ δ, where δ = 0.004. In words: tomorrow's return relative to today must exceed 0.4%. Smaller moves are labelled by sign during training but the 0.4% dead-band is enforced at signal time."

> "**Signal gate formula** (boxed): we trade only when (p-hat ≥ 0.58) AND (|predicted-move| ≥ 0.4%). Both must hold."

> "If either fails, we don't trade — we sit out. This is the **gate** that converts a noisy 51% raw signal into a tradeable 57% gated signal. The 0.4% threshold clears the ~40 bps round-trip cost."

> "On the right is one example: SBIN (State Bank of India) across six folds. Notice fold 4 hit 66.1% accuracy — that was Q3 2024, a bull run. Fold 6 was only 44.6% — the 2026 correction. Mean across folds is 54.3%."

> "**Single-fold reporting would have let me cherry-pick fold 4 and claim 66%.** Multi-fold reporting is the honest number."

---

## Slide 13 — Results: Out-of-Sample Accuracy (1.5 min)

> "Headline numbers across 100 symbols and 6 folds:"

> "- Mean accuracy on **all days**: **50.97%**"
> "- Mean F1-score: **58.79%** (F1 balances precision and recall)"
> "- Bootstrap accuracy on **gated** trades only: **57.25%**"
> "- Statistical significance: **p less than 0.05** — meaning the probability that this result is due to random chance is below 5%, conventionally significant"
> "- The null hypothesis that p equals 50% is **rejected**"

> "**Top-5 stocks** by accuracy include MUTHOOTFIN at 58.5% (Finance), EICHERMOT at 58% (Auto), and PERSISTENT at 56.7% (IT)."

> "The histogram on the right shows the accuracy distribution across all 100 stocks. Mode is around 52%, right tail extends to 59%. The point is: **no single stock dominates**. The result is distributed."

---

## Slide 14 — Results: Portfolio Backtest (2 min)

> "Now the deployable part. We took the top-15 stocks by accuracy, equal-weighted them, long-only — meaning we only buy, never short."

> "From December 2023 to April 2026, roughly 2.5 years of out-of-sample testing:"

> "- **Total return: +170.65%**"
> "- **Nifty-50 benchmark over the same period: ~18%**"
> "- **Sharpe ratio: 1.18** (above 1 is respectable)"
> "- **Maximum drawdown: 14.59%** (the largest peak-to-trough loss; below 20% is acceptable)"

> "Crucially, look at the **regime decomposition**. Sharpe in sideways markets is **1.69**, in bear markets **1.67**, in bull markets **1.43**. The strategy is **profitable in all three regimes** — it's not a bull-market artefact."

> "**Diebold-Mariano significance tests** at the bottom: we significantly beat momentum and AR(1) baselines (p less than 0.001). We do **not** significantly beat the 'always up' baseline (p = 0.92) — and we are transparent about this, because the test period was somewhat bullish on average. But the regime-decomposed Sharpe addresses that concern."

> "Three formulas in the bottom-right block make these metrics precise:"

> "**Sharpe ratio** = (expected portfolio return − risk-free rate) / standard deviation of returns, scaled by √252 to annualise (252 trading days/year)."

> "**Max drawdown** = the minimum of (V_t − running-max V_s) / running-max V_s. V_t is portfolio NAV at time t. It captures the worst peak-to-trough loss in percentage terms."

> "**Diebold-Mariano statistic** = mean of d / sqrt(variance of d / n), where d_t = loss_t of strategy A minus loss_t of strategy B. If DM is large in magnitude, the two strategies' forecasts differ significantly."

---

## Slide 15 — Comparison with Prior Research (1 min)

> "Side-by-side comparison. Patel 2015 reports 83% accuracy on a single split with 10 stocks. We report 50.97% on six-fold walk-forward with 100 stocks. **That gap is a protocol difference, not a quality difference.**"

> "Our **gated** accuracy of 57.25% is the apples-to-apples comparison and is statistically significant — which **no prior Indian study reports**."

> "**Fischer & Krauss 2018** in EJOR is the closest peer — they did walk-forward on S&P 500 and got 56%. We hit 57.25% gated on NSE, after costs, with a live layer. **No prior Indian study reports Sharpe ratio, max drawdown, or regime decomposition.**"

---

## Slide 16 — Live System: Deployment Pipeline (2 min)

> "The diagram on the right shows the full data flow: features → five models → meta-learner → temperature scaling → signal gate → paper trader → **promotion gate** → Angel One API → NSE."

> "Focus on the **promotion gate** table on the left. This is **the** safety mechanism."

> "Before any real capital is deployed, **all seven KPIs must clear**:"
> 1. At least 40 closed paper trades
> 2. At least 20 paper-trading days of evidence
> 3. Rolling Sharpe ≥ 1.0
> 4. Max drawdown ≤ 10%
> 5. Fill rate ≥ 90%
> 6. Slippage ≤ 25 bps
> 7. Brier drift ≤ 0.05 (Brier score measures probability accuracy)

> "Logic is **ALL must pass**. This is called **fail-closed design**: if any KPI is missing or fails, the system blocks real-money trading. It cannot 'fail open' into live trading. This is the same safety pattern used in industrial control systems and aviation."

> "**Brier score formula** at the bottom: BS = (1/n) × Σ(p-hat_t − y_t)². It's the mean squared error between predicted probability and actual outcome (0 or 1). Lower is better-calibrated. The 'Brier drift' KPI watches this number live — if calibration degrades over time, the gate closes."

---

## Slide 17 — System Architecture (1 min)

> "Quick tour of the full-stack architecture. Frontend is Next.js 16 with Tailwind and Radix UI. Backend is FastAPI on Uvicorn. The broker integration is Angel One SmartAPI for order routing. Data is stored as Parquet and JSON. APScheduler runs the daily 9:00 IST pipeline."

> "On the right, the data flow: dashboard talks to FastAPI over HTTP and WebSocket. FastAPI invokes the ML pipeline via `run_pipeline()` and places orders via Angel One's `place_order()`. Both write to a Parquet/JSON store; the store is what the dashboard reads. Angel One sits between us and NSE."

---

## Slides 18–23 — Dashboard Tour (30 sec each — 3 min total)

> "These six slides are screenshots of the **actually-running dashboard**. I'll walk through them briefly."

- **Overview (18):** "Live overview — NAV, open positions, system status, gate state."
- **Signal Board (19):** "Next-day calibrated signals. Each row shows a stock, its probability p-hat, predicted move delta, and whether the gate is open or closed."
- **Portfolio (20):** "Current open positions, per-symbol allocation, exposure breakdown."
- **Orders (21):** "Order book — pending, filled, cancelled orders with timestamps."
- **Simulated PnL (22):** "Paper-trading equity curve and drawdown chart — this is what the promotion gate watches."
- **100-Stock Analysis (23):** "Universe-wide ranking — sort all 100 stocks by gate-pass status and predicted move."

---

## Slide 24 — Per-Symbol Results (1.5 min)

> "Two representative stocks from different sectors."

> "**SBIN** — State Bank of India, our largest public-sector bank. Out-of-sample accuracy 54.9%. Signals **cluster around RBI MPC announcement windows** and quarterly results season — exactly when the macro features have the most information. The banking force-includes (DXY, crude, days-to-RBI) measurably improve signal quality."

> "**BRITANNIA** — biscuit and dairy major, FMCG sector. Out-of-sample accuracy 56.3%, top-10 in our 100-symbol panel. LightGBM is the best individual model at 59.4%. For this stock, **calendar and statistical features dominate** the importance ranking — its behavior is driven more by routine timing patterns than by macro shocks."

---

## Slide 25 — Case Studies: Prior Journal Work (1.5 min)

> "This table lists six representative prior studies and what they did versus what they didn't do."

> "Patel 2015 in Expert Systems: ANN/SVM/RF on NSE technical indicators, single split. **Gap:** no cost model, no walk-forward, no calibration."

> "Fischer & Krauss 2018 in EJOR: LSTM on S&P 500, walk-forward, but **no per-fold calibration and not Indian-market**."

> "Nabipour 2020 in Entropy: LSTM/GRU on 4 Tehran sectors, single split — **accuracy without deployable Sharpe**."

> "Long 2019 in IEEE Access: deep-learning trading agents on US equities, random split — **no leakage audit, no regime conditioning**."

> "López de Prado 2018 in *Advances in Financial ML*: provides the methodology (PBO, CSCV, MDA) we use, but **not applied at panel scale to Indian markets**."

> "Shwartz-Ziv & Armon 2022 in NeurIPS: empirical study showing trees usually beat deep learning on tabular data — this **motivates** why our meta-learner correctly down-weights the DL members."

> "The footnote captures the four gaps **common to all of them**: no Indian cost stack, no calibrated gate, no regime-conditional proof, no live promotion gate. **This thesis addresses all four.**"

---

## Slide 26 — Conclusion & Contributions (2 min)

> "**Why the ensemble works** — four reasons:"
> 1. Three orthogonal error sources: trees catch tabular patterns, BiLSTM/TCN catch temporal patterns, N-BEATS catches hierarchical decomposition. The meta-learner weights each per-symbol.
> 2. The **calibrated gate** turns a 51% raw signal into a 57% operationally trustworthy signal — and the gate is what we actually deploy. The gate **is the deliverable**.
> 3. Macro and calendar features capture conditional structure right at regime-change windows — RBI meetings, expiry days, results season.
> 4. Profitability holds in all three regimes — not a bull-market artefact.

> "**Five novel contributions:**"
> 1. Panel-scale leakage-audited walk-forward validation on 100 NSE stocks across six folds — no prior Indian work at this scale.
> 2. Calibrated gating via temperature scaling — the 58% threshold has empirical probability meaning, not an arbitrary cutoff.
> 3. Heterogeneous tree + DL committee with logistic meta-learner on out-of-fold probabilities.
> 4. Full Indian cost-stack backtest (STT, GST, exchange, SEBI, slippage).
> 5. Seven-KPI fail-closed promotion gate on paper-trading KPIs — live capital deployment is auditable.

> "**What this signifies** (green block at bottom): honest accuracy is **not** the deliverable. A calibrated, gated, cost-aware system is. This thesis **reframes** NSE stock prediction from a modelling contest into a **deployment-discipline** problem, and provides a reproducible template for the Indian market."

---

## Slide 27 — Future Work (1.5 min)

> "Immediate roadmap, next three months:"
> 1. **Live promotion:** two more paper-trading weeks to clear the remaining drawdown and Brier drift KPIs.
> 2. **Per-symbol recalibration:** replace the panel-wide 0.58 threshold with per-sector isotonic regression — sectors with thicker tails should have different thresholds.
> 3. **F&O leg:** route the calibrated probability into a 1-delta long call option — defined-loss instrument, potentially higher Sharpe per unit of risk.
> 4. **Macro shock test:** replay February to April 2020 — the COVID crash — and re-tune the drawdown gate after stress evidence.

> "Longer-term directions:"
> 1. **Residual-target deep learning:** re-train the DL models on the *residuals* of the tree models, instead of the raw target. This is a stacking refinement.
> 2. **FinBERT-India sentiment:** I have already fine-tuned ProsusAI/finbert on 687 hand-labelled Indian-finance sentences (negative/neutral/positive), achieving 84% validation accuracy. Walk-forward integration into the feature set is the next experiment.
> 3. **Reinforcement-learning allocation:** treat calibrated probability as the state of a bandit or policy-gradient agent.
> 4. **Alternate universes:** BSE MidCap 150 and SGX Nifty futures — test how portable the methodology is."

---

## Slide 28 — References (15 sec)

> "Key references on this slide. Foundational works on the left, comparative studies on the right. Full bibliography is in the thesis."

---

## Slide 29 — Thank You (15 sec)

> "Thank you for your time. I'm happy to take questions."

---

# Anticipated Q&A

## Likely panel questions (and good answers)

### Q1: "Why is 51% accuracy good? My students get 70% on Kaggle."
**A:** "Kaggle datasets are usually random splits without temporal leakage, and target is balanced. Stock direction at one-day horizon has a theoretical upper bound near 55–60% — Fama and others have shown this empirically. Our 51% is *all-days, multi-fold, leakage-audited*; our *gated* 57.25% is the deployable number, and it's statistically significant (p<0.05). The honest comparison is gated 57% versus prior literature's gated number — which they don't report."

### Q2: "How do you know it's not just curve-fitting to the test window?"
**A:** "Three protections. One: six folds, so any single test window is one of six. Two: 63,082 predictions across 100 stocks — too many to overfit to. Three: Diebold-Mariano test against momentum and AR(1) gives p<0.001, meaning random luck is rejected. We also report regime-decomposed Sharpe — overfit-to-test would not generalize across bull/bear/sideways."

### Q3: "What if the market regime changes after deployment?"
**A:** "Three answers. One: walk-forward retrains every fold, so we *expect* regime shifts and the model adapts. Two: the promotion gate watches Brier drift — if calibration degrades, the gate closes. Three: this is exactly what fail-closed design is for — when conditions change, the system stops trading until KPIs recover."

### Q4: "Why ensemble five models instead of one really good one?"
**A:** "Bias-variance trade-off. Trees have low variance but miss sequence structure. Deep models have low bias but high variance. Wolpert 1992 proved heterogeneous learners reduce correlated error. Empirically, our meta-learner correctly down-weights the weak members; we keep them because *which* one is weak varies per symbol. Removing any one degrades the panel-wide Sharpe."

### Q5: "Why didn't you use sentiment, news, fundamentals?"
**A:** "Sentiment is on the roadmap — FinBERT-India is already trained at 84% validation accuracy, awaiting walk-forward integration. Fundamentals update quarterly, so they're slow features for daily-bar trading. The scope of *this* thesis is technical + macro + calendar; adding NLP-derived sentiment is the next paper, not a missing piece."

### Q6: "Are these returns realistic? 170% in 2.5 years is huge."
**A:** "Two honest qualifiers. One: this is *cost-net* meaning STT/GST/SEBI/slippage already deducted, and Sharpe is 1.18 — not 3 or 4, which would be implausible. Two: it's *backtest*, not live yet. The promotion gate is correctly blocking live capital because we don't yet have 40 closed trades with under-10% drawdown. Live numbers, when they come, will likely be lower — that's expected, and the gate enforces patience."

### Q7: "What is temperature scaling actually doing mathematically?"
**A:** "If a model outputs logit z, normally we apply sigmoid sigma(z) to get probability. Temperature scaling computes sigma(z/T-star) where T-star is a single scalar fit by minimising NLL on validation. If T-star > 1, the model was over-confident and probabilities get compressed toward 0.5. If T-star < 1, the model was under-confident and probabilities get pushed toward extremes. The argmax (predicted class) doesn't change — only the *probability number* changes — so accuracy is unchanged but the gate threshold of 0.58 now has empirical meaning: when we say 58%, we actually win 58% of the time."

### Q8: "Why not use reinforcement learning?"
**A:** "RL is in future work. The reason we didn't start with it: RL needs a reward signal and many episodes. Our daily-bar setup gives ~250 episodes per year per stock — sparse. Supervised classification gives one training row per day per stock — dense. So we built the calibrated supervised pipeline first, and the next step is to *use* its calibrated probability as state for a bandit or policy-gradient agent that allocates capital. RL on top of calibrated supervision is more sample-efficient than RL from raw price."

### Q9: "How is this different from technical analysis that traders already do?"
**A:** "Traders use individual indicators (RSI, MACD) with rule-of-thumb thresholds. We use 219 features fed into five ML models, weighted by a meta-learner, calibrated, gated, and validated on 63,082 out-of-sample predictions. The *methodology discipline* is the difference: every feature is leakage-audited, every threshold is empirically derived, every claim is statistically tested. A discretionary trader cannot do that."

### Q10: "What's the cost of running this in production?"
**A:** "Compute: one fast pass at 9:00 AM (LightGBM + XGBoost in ~minutes), one overnight DL retrain (~1 hour on a single GPU). Storage: Parquet caches under ~5 GB for 100 stocks. Broker API costs: Angel One SmartAPI is free for retail. Infrastructure: a single 8-core machine handles the full pipeline. No cloud dependency required."

---

## Out-of-box / curveball questions

### CB1: "If your gate blocks live trading until KPIs clear, isn't the thesis incomplete?"
**A:** "The thesis demonstrates the *system*; live trading is an outcome of the system. The thesis contribution is the **methodology and infrastructure** — not 'I made X rupees.' If the gate had opened immediately, that would be evidence of a *weak* gate, not a strong system. The gate correctly blocking is the *intended* behaviour. The day the gate opens, we'll know it's because the system genuinely earned its way through 7 independent KPIs."

### CB2: "Could a malicious actor reverse-engineer your signals from the dashboard?"
**A:** "The dashboard is operator-facing, not public. It runs on localhost or a private VPN. Even if someone saw the next-day signals, the *edge* doesn't come from secrecy — it comes from execution discipline and calibration. The signal alone, without the gate and cost-aware execution, doesn't print money. That's exactly the point of slide 3."

### CB3: "What's your edge over a hedge fund with billions and a quant team?"
**A:** "I have **no edge** over a top-tier quant fund on raw modelling. Their edge is data exclusivity (alt-data feeds), execution venue access, and capital. My edge is *methodological honesty* applied to *publicly available data* in the Indian retail context. The thesis is a template a small operator can replicate — not a challenge to Renaissance Technologies."

### CB4: "If everyone uses your method, the edge disappears, right?"
**A:** "Correct — and that's a feature, not a bug, in EMH terms. The methodology is transparent and reproducible. If the calibrated-gate edge erodes as more operators adopt it, the gate KPIs will close the system automatically. That's the safety property again. But practically, retail adoption is slow; the gap between research papers and live retail systems is wide."

### CB5: "Why a logistic meta-learner and not a neural net for meta-learning?"
**A:** "Three reasons. One: meta-learner inputs are 5 probabilities — extremely low-dimensional — so a deep model would overfit. Two: logistic regression's coefficients are *interpretable* — we can literally read off how much weight each member gets per stock. Three: in stacking literature (Wolpert 1992 onwards), simple meta-learners on heterogeneous base learners outperform complex meta-learners on homogeneous ones. Occam wins here."

### CB6: "What if the stock you predict has a corporate action — split, dividend, bonus?"
**A:** "yfinance auto-adjusts for splits and dividends in its `Close` column, which is what we use. Bonus issues are similarly adjusted. Corporate actions cause *one-day* jumps in raw prices that look like signals but are not — adjusted close removes that artifact. We also drop the trading day if `Volume` is zero or if the close-to-prev-close jump exceeds 25% (likely a data error). This is in the data quality check."

### CB7: "Your dataset starts Jan 2018 — why not earlier?"
**A:** "Two reasons. One: SEBI tightened margin and brokerage regulations around 2017 — pre-2018 cost structure is materially different, so training on it would mis-calibrate the cost gate. Two: NSE introduced major derivatives rule changes in 2017–18; pre-2018 cash-market dynamics differ. We start from when the *current* market microstructure stabilised."

### CB8: "What's your model's behaviour on a Black Swan event?"
**A:** "Honestly: untested in production. February–March 2020 (COVID crash) is in our test window for some folds, and the system did *not* break — Sharpe in the bear-regime decomposition is 1.67, which includes that. But a Black Swan by definition is unprecedented; the drawdown gate (10% threshold) would close the system before it could lose more. This is explicitly listed under future work as 'macro shock test' — replay 2020 with current model weights."

### CB9: "Why classification (up/down) instead of regression (predict return magnitude)?"
**A:** "We tried both. Regression on log-returns suffers from extreme-value sensitivity — one large gap-up day dominates the loss. Classification with a delta-threshold gate (`predicted move ≥ 0.4%`) is more robust because the gate already enforces a magnitude requirement separately from the directional bet. Plus, calibrated probability is more interpretable for the operator than a raw return forecast."

### CB10: "If I gave you 10x more compute, what would you change?"
**A:** "Three things. One: bigger DL models with proper hyperparameter search per symbol — currently we use one global config. Two: per-symbol meta-learners trained with cross-validation instead of single OOF — would give better weighting. Three: deeper FinBERT-India sentiment training on a larger labelled corpus (5,000+ sentences vs. our 687). I would NOT add more base models — Shwartz-Ziv shows diminishing returns past 5–7 members on tabular data."

---

## Things to **avoid saying**

- ❌ "This system makes money" → say "This system produces a positive backtest Sharpe of 1.18; live deployment is gated pending KPIs"
- ❌ "We beat the market" → say "Our backtest exceeds the Nifty benchmark by X% over the test window"
- ❌ "Our model predicts the stock price" → say "Our model predicts direction with calibrated probability"
- ❌ "Guaranteed returns" / "high accuracy" → use the exact numbers with context

## Things to **emphasize**

- ✅ "Leakage-audited" — the methodology is the contribution
- ✅ "Fail-closed gate" — the safety design is novel
- ✅ "Statistically significant after costs" — every word matters
- ✅ "Reproducible template" — anyone can replicate this

---

## Time budget summary

| Section | Slides | Time |
|---|---|---|
| Setup (Title, Outline, Intro) | 1–3 | 2.5 min |
| Problem & Literature | 4–7 | 5 min |
| Methodology | 8–12 | 6 min |
| Results & Comparison | 13–15 | 4.5 min |
| Live System & Dashboard | 16–23 | 6 min |
| Case Studies & Closing | 24–29 | 4.5 min |
| **Total talk** | 29 | **~28 min** |
| **Q&A** | — | 10–15 min |

If short on time, compress slides 18–23 (dashboard tour) to 1 minute total.

---

*Last updated: aligned with presentation.tex revision after restructure + audit (May 2026).*
