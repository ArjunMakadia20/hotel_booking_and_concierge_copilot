# Hotel Booking Cancellation Prediction — Presentation Script
### 25–30 minute technical + business walkthrough
*Audience: shareholder / hotel executive (limited ML background), with your supervisor able to interrupt with technical questions.*

Every number in this script is pulled directly from the repo: `reports/classification_model_comparison.csv`, `reports/classification_reports.md`, `reports/eda_summary.md`, `reports/modeling_concepts.md`, `reports/outlier_report.csv`, the plots in `reports/`, the MLflow runs logged by `src/train.py`, and live output from `src/predict.py`. Nothing here is invented.

**How to use this doc:** each section has *Slide bullets* (put these on screen), a *Say this* script (speak this, don't read it verbatim — know it well enough to paraphrase), and sometimes an *If asked* box for technical follow-ups. Rough timings add up to ~28 minutes; adjust pacing live based on how many questions land.

---

## 0. Opening (30 sec)

**Say this:** "Today I'm walking through a machine learning system we built to predict whether a hotel booking will be cancelled, and how confident the model is about that prediction. I'll cover the business case, the data, how we built and compared five different models, the one we selected, and how it's deployed. I'll pause for questions, but feel free to jump in anytime."

---

## 1. Executive Summary (2 min)

**Slide bullets:**
- Problem: hotels don't know in advance which bookings will cancel
- Solution: a model that predicts cancellation + gives a probability (e.g. "92% likely to cancel")
- Built and compared 5 ML models on 119,390 real bookings
- Best model: XGBoost — 88.2% accuracy, 95.4% ROC-AUC
- Deployed two ways: batch (offline) and real-time (online API)

**Say this:** "Every hotel deals with cancellations — but today, a hotel only finds out a room is available again *after* the guest cancels, often too late to resell it at full value. That's lost revenue and wasted forecasting.

What we built is a system that looks at booking details — how far in advance it was made, the deposit type, the guest's history — and predicts, before the fact, how likely that booking is to cancel. Not just yes or no: a probability, like 'this booking is 92% likely to cancel.' That's actionable. A hotel could follow up with a phone call, adjust overbooking strategy, or release the room proactively.

**Business value:** if we can flag high-risk bookings early, hotels can rebook, overbook more confidently within safe limits, tailor deposit policies for risky segments, and get more accurate revenue forecasts. Even a modest reduction in unexpected vacant-room nights translates directly into recovered revenue — and this doesn't require new infrastructure, just plugging a prediction into existing front-desk or revenue-management workflows."

**If asked — "What's the ROI, concretely?"**
"We haven't run a live financial pilot yet, so I won't invent a dollar figure. What I can say concretely: the model correctly catches about 81% of bookings that go on to cancel [recall = 0.813], with 86% precision, meaning when it flags a booking as high-risk, it's right the large majority of the time. That combination is what makes early intervention worthwhile rather than noisy."

---

## 2. Dataset (3 min)

**Slide bullets:**
- 119,390 bookings, 32 original columns
- Source: two hotels (City Hotel, Resort Hotel), historical booking records
- Target: `is_canceled` (0 = stayed, 1 = cancelled)
- Two leakage columns removed before modeling

**Say this:** "Our dataset has 119,390 real bookings across two hotel types — a City Hotel and a Resort Hotel — with 32 columns describing each booking: when it was made, who booked it, how they paid, and what happened.

**The target variable** is `is_canceled` — a simple 0 or 1. 0 means the guest showed up (or the booking wasn't cancelled), 1 means it was cancelled. That's what we're predicting, along with the probability behind it.

**The most important features, and why each one is useful:**
- **`lead_time`** — days between booking and arrival. Intuition: the further out someone books, the more time for plans to change.
- **`deposit_type`** — No Deposit / Non Refund / Refundable. This turned out to be our single strongest signal — I'll show why in a moment.
- **`market_segment` / `distribution_channel`** — how the booking arrived (Direct, Corporate, Online Travel Agency, etc.). Different channels attract different guest commitment levels.
- **`previous_cancellations`** — has this guest cancelled with us before? Past behavior predicts future behavior.
- **`total_of_special_requests`** and **`required_car_parking_spaces`** — proxies for how invested the guest is in the specific stay.
- **`adr`** (average daily rate), **`adults`/`children`/`babies`**, **`stays_in_weekend_nights`/`stays_in_week_nights`** — booking economics and party size.
- **`customer_type`**, **`reserved_room_type`** / **`assigned_room_type`**, **`country`**, **`agent`/`company`** — guest and booking-channel identifiers.

We also removed two columns before modeling: `reservation_status` and `reservation_status_date`. I'll explain why in the next section — it's an important detail."

**If asked — "Why two hotels together, not separate models?"**
"We kept them together and let `hotel` be a feature, since we wanted one deployable model; the EDA plot `cancellation_rate_by_hotel_type.png` shows City Hotel cancels more than Resort Hotel, and the model learns that distinction automatically rather than needing two separate pipelines."

---

## 3. Data Cleaning (2.5 min)

**Slide bullets:**
- Missing values: `children`→0, `country`→"UNKNOWN", `agent`/`company`→0
- **Leakage removed:** `reservation_status`, `reservation_status_date`
- Categorical encoding: one-hot (rare categories bucketed, unseen categories ignored)
- **No normalization** — by design
- Stratified 80,000 / 39,390 train/test split

**Say this:** "Before modeling, four things happened to the raw data:

**Missing values** — very few, and all handled sensibly: `children` had 4 missing values, filled with 0. `country` had 488 missing (0.4%), filled with an explicit 'UNKNOWN' category. `agent` and `company` are booking-system ID codes; `company` specifically is missing 94% of the time, so we fill both with 0, meaning 'no agent/company on file' — that's meaningful information itself, not something to throw away.

**Data leakage — this is the most important cleaning decision.** Two columns, `reservation_status` and `reservation_status_date`, record the *final outcome* of the booking — one of the values of `reservation_status` is literally 'Canceled.' If we left that in, the model would just be looking up the answer, and it would be useless in the real world because that field doesn't exist until *after* we already know what happened. We removed both before training.

**Encoding** — categorical columns like `deposit_type` and `market_segment` get one-hot encoded, turning each category into its own 0/1 column. Rare categories get bucketed together so the encoding doesn't explode, and any brand-new category at prediction time is safely ignored rather than crashing.

**No normalization** — by explicit design. Tree-based models like Random Forest and XGBoost don't need scaled numbers, so scaling here would add complexity for no benefit. The one exception is the neural network (MLP), which is scale-sensitive — we only scale features inside its own internal pipeline, nowhere else.

**Train/test split** — stratified so cancellation rate is preserved in both sets, using 80,000 bookings for training and the remaining 39,390 for testing — a split large enough to trust the results."

**If asked — "Why not drop `company` given it's 94% missing?"**
"Missingness itself is a signal — most individual leisure bookings simply don't have a company code, so 'has a company code or not' is informative, not noise. Dropping it would throw away that signal for no accuracy gain."

---

## 4. Exploratory Data Analysis — every graph (7–8 min)

**Say this (intro):** "Before building any model, we ran a full exploratory analysis — 17 visualizations. I'll go through each one and what it tells us."

> Speaker note: don't read every bullet aloud for every graph — hit "what it shows" and "business insight" for most, and slow down on deposit_type, lead_time, and correlation heatmap since those carry the presentation's central argument.

**1. `cancellation_class_distribution.png`**
- Shows: 75,166 bookings not cancelled (63.0%) vs 44,224 cancelled (37.0%).
- Why: this is simply how often people cancel across this dataset.
- Insight: cancellation is common enough to matter financially, but not so imbalanced that the model can't learn — it's a solvable problem.
- Business use: sets a baseline — any model must clearly beat "assume 37% might cancel" to be useful.

**2. `cancellation_rate_by_hotel_type.png`**
- Shows: City Hotel cancels more often than Resort Hotel.
- Why: resort stays tend to be pre-planned vacations with more commitment (flights, time off booked); city stays are often more flexible business/short trips.
- Insight: hotel type alone is a mild risk signal.
- Business use: City Hotel could apply slightly stricter deposit policies.

**3. `cancellation_rate_by_lead_time.png`**
- Shows: overlapping histograms — cancelled bookings skew toward longer lead times than completed stays.
- Why: more time between booking and arrival means more time for plans to change.
- Insight: last-minute bookings are safer; far-out bookings carry more uncertainty.
- Business use: apply more flexible policies to last-minute bookings, firmer ones to advance bookings.

**4. `cancellation_rate_by_lead_time_group.png`**
- Shows: cancellation rate rises step by step across lead-time buckets (0–7 days, 8–30, 31–90, 91–180, 181–365, 365+).
- Why: makes the pattern in graph 3 concrete and monotonic — the relationship is consistent, not noisy.
- Insight: lead time is a genuinely reliable, gradually-increasing risk indicator.
- Business use: could tier deposit requirements by lead-time bracket.

**5. `cancellation_rate_by_arrival_month.png`**
- Shows: cancellation rate varies by arrival month (seasonality).
- Why: demand patterns, holidays, and weather-dependent travel plans shift booking commitment across the year.
- Insight: some months are structurally riskier for no-shows/cancellations.
- Business use: staff overbooking buffers seasonally instead of using one flat policy year-round.

**6. `cancellation_rate_by_market_segment.png`**
- Shows: booking channels like Online TA and Groups cancel more than Direct or Corporate.
- Why: direct and corporate bookings usually have more accountability (a company or a direct relationship with the hotel); online-agency bookings are often price-shopped across multiple sites with no penalty for booking several "just in case."
- Insight: acquisition channel predicts commitment level.
- Business use: revenue management could weight channel risk into pricing/overbooking decisions.

**7. `cancellation_rate_by_distribution_channel.png`**
- Shows: similar story to market segment — TA/TO channels cancel more than Direct/Corporate/GDS.
- Why: overlaps conceptually with market segment (how the booking physically arrived at the hotel).
- Insight: reinforces channel-based risk, from a slightly different angle (booking pipeline vs. marketing category).
- Business use: same as above — channel-aware policy.

**8. `cancellation_rate_by_deposit_type.png`** — *the single most important graph in this deck*
- Shows: **Non-Refund deposits cancel 99.4% of the time**, vs 28.4% for No Deposit and 22.2% for Refundable.
- Why: this is a known quirk of this specific dataset — Non-Refund bookings here are disproportionately associated with high-cancellation patterns (likely tied to specific booking channels/agents in the source data), not a universal real-world rule that "non-refundable = definitely cancels."
- Insight: this single feature ends up explaining roughly **60% of the model's total decision-making** (more in Section 10).
- Business use: **flag this explicitly as a data caveat, not a policy recommendation** — I'll say clearly to your supervisor that we should NOT conclude "stop offering non-refundable rates." It reflects this dataset's specific booking mix, and needs real-world validation before acting on it commercially.

**9. `cancellation_rate_by_customer_type.png`**
- Shows: Transient (individual, no group affiliation) bookings cancel somewhat more than Contract or Group types.
- Why: individual travelers have fewer commitments (no group coordination, no contract terms) tying them to the booking.
- Insight: booking "type" carries real signal about commitment.
- Business use: contract/negotiated-rate customers are lower cancellation risk — can be treated more leniently on deposits.

**10. `cancellation_rate_by_reserved_room_type.png`**
- Shows: cancellation rate varies somewhat by which room type was originally reserved.
- Why: certain room types may correlate with certain guest segments or price points that carry different commitment levels.
- Insight: minor but present signal.
- Business use: minor input to risk scoring — not something to act on standalone.

**11. `cancellation_rate_by_assigned_room_type.png`**
- Shows: similar pattern to reserved room type, using the room actually assigned.
- Why: closely related to #10; differences between reserved vs. assigned type can also reflect operational reassignment.
- Insight: adds a little more resolution beyond the original reservation.
- Business use: same as #10 — supporting signal, not a standalone driver.

**12. `cancellation_rate_by_special_requests.png`**
- Shows: more special requests → lower cancellation rate.
- Why: a guest who asks for a late check-in, extra pillows, or a specific view has mentally committed to the trip — they're planning the details.
- Insight: engagement/specificity of a booking predicts follow-through.
- Business use: bookings with zero special requests could be flagged as slightly higher risk for a courtesy confirmation call.

**13. `cancellation_rate_by_previous_cancellations.png`**
- Shows: guests with prior cancellations cancel again at a much higher rate.
- Why: cancellation is behavioral — people who've done it before are more likely to do it again (booking speculatively, changing plans often, etc.).
- Insight: guest history is a strong, simple risk signal.
- Business use: loyalty/CRM systems could flag repeat-cancellers for different deposit terms.

**14. `cancellation_rate_by_parking_spaces.png`**
- Shows: needing a car parking space correlates with a much lower cancellation rate.
- Why: same logic as special requests — planning specific logistics signals real intent to arrive.
- Insight: another "commitment" proxy.
- Business use: low-effort risk signal that's already collected at booking time.

**15. `missing_values.png`**
- Shows: only `company` (94.3%), `agent` (13.7%), `country` (0.4%), and `children` (~0%) have missing data; everything else is complete.
- Why: this is a clean, well-maintained hotel reservation system — most fields are mandatory at booking time.
- Insight: minimal data-quality work was needed; we didn't have to discard large chunks of data.
- Business use: reassures stakeholders the underlying booking system captures reliable data.

**16. `correlation_heatmap.png`**
- Shows: numeric-feature correlation matrix, ordered by strength of relationship with `is_canceled`. The strongest numeric correlate is `lead_time` at only 0.293 — no numeric feature exceeds 0.30.
- Why: this is a key finding — cancellation isn't explained by any single number rising or falling. It's explained by *combinations* (e.g., long lead time **and** a certain deposit type **and** a certain channel).
- Insight: this is our first concrete evidence that **the relationship is non-linear** — a simple straight-line model can't capture it well.
- Business use: justifies why we tested tree-based/ensemble models rather than relying on a simple scoring formula.

**17. `outlier_boxplots.png`**
- Shows: boxplots for 13 numeric fields, flagging statistical outliers via the IQR method (full numbers in `outlier_report.csv`).
- Why: fields like `adults` (24.9% flagged), `booking_changes` (15.1%), `children` (7.2%) have "outlier" values by strict statistical definition — but these are legitimate rare bookings (large families, heavily modified reservations), not data errors.
- Insight: we deliberately **did not remove** these — tree-based models handle outliers well, and these tails carry real signal (e.g., extreme `booking_changes` might indicate an unstable, cancellation-prone booking).
- Business use: shows the data was audited carefully rather than blindly scrubbed, which matters for trusting the model's conclusions.

---

## 5. Statistical Patterns (2 min)

**Slide bullets:**
- Strongest relationship: `deposit_type` (Non-Refund → 99.4% cancel)
- Second cluster: `lead_time`, `market_segment`, `country`
- Weakest: party size, meal plan, most date fields
- Surprise: no single numeric feature is strongly linearly correlated with cancellation

**Say this:** "Pulling it together: the strongest relationship in the entire dataset is `deposit_type` — it dominates everything else. The next tier is booking-channel and lead-time related: `market_segment`, `distribution_channel`, `country`, and `lead_time` itself. Guest-history features like `previous_cancellations` and engagement signals like `total_of_special_requests` and `required_car_parking_spaces` form a third, smaller-but-real tier.

**What surprised us:** going in, we expected `lead_time` or `adr` to dominate, since those are the classic 'obvious' predictors in hospitality analytics. Instead, no single *numeric* feature correlates strongly on its own — the strongest numeric correlation is only 0.293. The real signal was hiding in a *categorical* feature and in *combinations* of features, which is exactly why a simple scorecard or spreadsheet formula wouldn't have caught this, but a machine learning model does."

---

## 6. Machine Learning Pipeline (2 min)

**Slide bullets:**
```
Dataset → Cleaning (leakage removal, missing values) → Encoding (one-hot)
→ Train/Test Split (80,000 / 39,390) → Train 5 Models → Compare
→ Select Best (XGBoost) → Save Model → Offline Batch Predictions (CSV)
→ Online API (FastAPI) → MLflow Experiment Tracking
```

**Say this:** "Here's the full pipeline end to end. We load the raw CSV, clean it and remove the leakage columns, one-hot encode the categoricals, split into 80,000 training and 39,390 test bookings, then train five different model types in parallel and compare them on the same test set. We pick the best one — XGBoost — save it to disk, and then it's usable two ways: batch scoring a whole file of new bookings offline, or a live API that scores one booking in real time. Every training run — including a follow-up hyperparameter experiment — is logged to MLflow so results are tracked and reproducible, not just remembered informally."

---

## 7. Explain Every Model (5–6 min)

**Slide bullets:**
| Model | Type | ROC-AUC |
|---|---|---|
| Logistic Regression | Linear | 0.874 |
| Decision Tree | Non-linear | 0.927 |
| Random Forest | Non-linear ensemble | 0.953 |
| MLP (deep learning) | Non-linear | 0.939 |
| XGBoost | Non-linear ensemble | **0.954** |

### Logistic Regression
**Say this:** "This is our linear baseline. It computes a weighted sum of every feature and squashes it through a curve into a probability between 0 and 1 — like a formula that says 'add up these weighted factors, and that total tells you the risk.'
- **Advantage:** simple, fast to explain, a fair baseline.
- **Disadvantage:** it can only add up independent effects — it can't say 'long lead time only matters *when combined with* a certain deposit type.' It misses interactions.
- **Why we trained it:** every classification project needs a linear baseline to prove the more complex models are actually earning their complexity.
- **Why it performed the way it did:** it scored lowest (87.4% ROC-AUC) precisely because cancellation is driven by feature interactions, which this model structurally cannot see."

### Decision Tree
**Say this:** "This model asks a sequence of yes/no questions — 'Is deposit Non-Refund? If not, is lead time over 150 days?' — and so on, splitting the data at each step.
- **Advantage:** naturally captures interactions between features (the second question only gets asked in context of the first); easy to visualize as a flowchart.
- **Disadvantage:** a single tree can memorize noise in the training data if grown too deep — that's overfitting. We capped its depth to keep it honest.
- **Why we trained it:** it's the natural next step up from a linear model, and the building block for the two ensemble models that follow.
- **Why it performed the way it did:** it jumped to 92.7% ROC-AUC, well above Logistic Regression, because it can finally see feature interactions — but it's still just one tree's opinion, less stable than an ensemble."

### Random Forest
**Say this:** "This builds 200 different decision trees, each trained on a random sample of the data and a random subset of features, then averages all their votes.
- **Advantage:** averaging many different trees cancels out each individual tree's mistakes — much more stable and accurate than one tree.
- **Disadvantage:** slightly slower and less interpretable than a single tree; still can't quite match boosting on this data.
- **Why we trained it:** the standard, reliable next step after a single tree — 'wisdom of the crowd' applied to trees.
- **Why it performed the way it did:** 95.3% ROC-AUC, almost tied with XGBoost — proof that averaging many trees substantially improves on one tree (92.7% → 95.3%)."

### MLP (Multi-Layer Perceptron — our deep learning model)
**Say this:** "This is a small neural network: layers of artificial neurons, each computing a weighted sum and passing it through a non-linear activation function, stacked to approximate very complex curved patterns.
- **Advantage:** in theory, can model almost any pattern given enough data; the most flexible model we tried.
- **Disadvantage:** needs feature scaling to train properly (the only model in this project we scale, and only inside its own pipeline); much slower to train (18.7 seconds vs about 1–2 seconds for the tree models); can overfit without care — we used early stopping to guard against that.
- **Why we trained it:** the supervisor's brief specifically asked for one deep learning technique, and it's a fair, modern comparison point against tree-based methods.
- **Why it performed the way it did:** 93.9% ROC-AUC — solid, but *not* the winner. On this kind of structured, spreadsheet-style tabular data, tree-based boosting methods like XGBoost typically outperform neural networks; neural nets tend to shine more on images, text, and audio. This is a genuinely useful, realistic finding, not a failure."

### XGBoost — the winner
**Say this:** "XGBoost also builds many trees, but sequentially: each new tree is trained specifically to fix the mistakes of the trees before it, with built-in penalties to stop any one tree from dominating.
- **Advantage:** typically the strongest performer on structured, tabular business data like ours; fast to train even with hundreds of trees; naturally captures feature interactions and non-linear thresholds.
- **Disadvantage:** more hyperparameters to tune than a single tree; slightly less immediately interpretable than a shallow tree (though we can still extract feature importance, shown in Section 10).
- **Why we trained it:** it's the industry-standard choice for exactly this kind of problem — tabular data with a mix of categorical and numeric features.
- **Why it performed the way it did:** it won on **every metric except pure precision** — 88.2% accuracy, 95.4% ROC-AUC, 83.6% F1 — while also being one of the *fastest* models to train (1.2 seconds). Boosting's error-correcting design let it capture the deposit-type/lead-time/channel interactions better than any other model we tried."

---

## 8. Model Comparison (2.5 min)

**Slide bullets — full comparison table:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.882** | 0.861 | **0.813** | **0.836** | **0.954** | 1.2s |
| Random Forest | 0.879 | **0.891** | 0.767 | 0.825 | 0.953 | 2.0s |
| MLP | 0.863 | 0.839 | 0.781 | 0.809 | 0.939 | 18.7s |
| Decision Tree | 0.848 | 0.838 | 0.732 | 0.782 | 0.927 | 0.9s |
| Logistic Regression | 0.806 | 0.798 | 0.638 | 0.709 | 0.874 | 14.3s |

**Say this:** "XGBoost wins on accuracy, recall, F1, and ROC-AUC — the four metrics that matter most for this project — and comes in second on training speed. Random Forest is genuinely close (0.953 vs 0.954 ROC-AUC) and actually has *better precision* — meaning fewer false alarms — but it misses more real cancellations (76.7% recall vs XGBoost's 81.3%). Since a missed cancellation costs the hotel more than a false alarm, we lean toward XGBoost's recall advantage.

The other three underperform for reasons we already covered: Logistic Regression can't see interactions, Decision Tree is a single unstable tree, and MLP — while capable — simply doesn't have the tabular-data advantage that boosted trees do, and it took 15 times longer to train for a worse result.

**On training time specifically:** Decision Tree is technically fastest (0.9s), but its accuracy is meaningfully lower. XGBoost, at 1.2 seconds, gives us near-best speed *and* the best accuracy — there's no real tradeoff here, which is part of why it's an easy recommendation."

---

## 9. Evaluation Metrics — explained for a hotel executive (5 min)

**Say this (intro):** "I want to walk through what each number in that table actually means in plain terms, using XGBoost's real results, tested on 39,390 real bookings we held back and never showed the model during training."

### Accuracy — 88.2%
- **Definition:** the percentage of all predictions that were correct.
- **Why it matters:** the simplest headline number.
- **Why 88.2% is good:** a hotel that just assumed "nobody cancels" would already be right 63% of the time — so 88.2% shows the model learned real patterns, not just the easy majority guess.
- **Hotel example:** out of every 100 bookings, the model gets about 88 right.
- **Caveat to mention:** accuracy alone can be misleading with imbalanced data — that's why we look at the next five metrics too.

### Precision — 86.1%
- **Definition:** of the bookings the model flagged "will cancel," how many actually did.
- **Why it matters:** controls false alarms.
- **Why 86.1% is good:** if front-desk staff act on a "high risk" flag (e.g., a courtesy call, releasing the room), they'll be right about 6 times out of 7.
- **Hotel example:** out of every 100 bookings flagged as risky, about 86 genuinely do cancel — only 14 false alarms.

### Recall — 81.3%
- **Definition:** of the bookings that actually cancelled, how many the model caught in advance.
- **Why it matters:** the most business-critical metric here — a **missed** cancellation (false negative) is the costly mistake, since the hotel loses the chance to resell that room.
- **Why 81.3% is good:** we catch roughly 4 out of every 5 real cancellations before they happen.
- **Hotel example:** if a hotel has 1,000 cancellations in a month, this model would have flagged about 810 of them in advance.

### F1-score — 0.836
- **Definition:** a single balanced score combining precision and recall.
- **Why it matters:** proves the model isn't gaming one metric at the expense of the other (e.g., flagging *everything* as risky just to maximize recall).
- **Why 0.836 is good:** precision (86.1%) and recall (81.3%) are close together — a model that cheated would show a lopsided gap between them.
- **Hotel example:** a well-rounded score confirms the alerts are both frequent enough and trustworthy enough to act on operationally.

### ROC-AUC — 0.954
- **Definition:** how well the model *ranks* bookings from riskiest to safest, across every possible decision threshold — not just the default 50% cutoff.
- **Why it matters:** since our output is a *probability*, not just yes/no, this measures whether the probabilities are actually meaningful and ordered correctly.
- **Why 0.954 is good:** 1.0 would be a perfect ranking, 0.5 is a coin flip — 0.954 is very close to perfect.
- **Hotel example:** if you lined up two bookings, one that ended up cancelling and one that didn't, the model correctly identifies which one was riskier about 95% of the time.

### Training Time — 1.2 seconds
- **Definition:** how long it takes to train the model on 80,000 bookings.
- **Why it matters:** not an accuracy metric — a *practicality* metric.
- **Why 1.2 seconds is good:** the model can be retrained weekly or even daily as new booking data comes in, at essentially no computing cost.
- **Hotel example:** the system can stay current with changing guest behavior (e.g. a new travel trend) without any expensive re-training project.

### Confusion Matrix (XGBoost, on the 39,390-booking test set)
| | Predicted: Not Cancelled | Predicted: Cancelled |
|---|---|---|
| **Actual: Not Cancelled** (24,799) | 22,879 (correct) | 1,920 (false alarm) |
| **Actual: Cancelled** (14,591) | 2,729 (missed) | 11,862 (correct catch) |

**Say this:** "This is the full breakdown behind every metric above. Of 24,799 bookings that didn't cancel, we correctly called 22,879 of them safe, with 1,920 false alarms. Of 14,591 bookings that did cancel, we correctly caught 11,862 of them in advance, and missed 2,729. In hotel terms: for every roughly 39,000 bookings, this model gives the revenue team advance warning on about 11,900 real cancellations, at the cost of about 1,900 unnecessary flags."

---

## 10. Feature Importance (2.5 min)

**Slide bullets — top contributors (aggregated by original column, from the trained XGBoost model):**

| Feature | Contribution |
|---|---|
| `deposit_type` | 59.9% |
| `market_segment` | 8.0% |
| `country` | 7.3% |
| `required_car_parking_spaces` | 5.0% |
| `customer_type` | 2.6% |
| `previous_cancellations` | 2.1% |
| *(all other 23 features combined)* | ~15% |

**Say this:** "This is what the model actually leans on to make its decisions — extracted directly from the trained XGBoost model, not guessed from correlation alone. Just four features — deposit type, market segment, country, and parking spaces — explain roughly 80% of the model's predictive power.

**deposit_type dominates at 59.9%** — I flagged this in the EDA section, and I want to repeat it here: this is a strong pattern *in this dataset*, and it's the main reason the model performs so well, but it also means the model's accuracy is somewhat concentrated on one feature. We should validate that this pattern generalizes to a hotel's live booking data before making commercial decisions based on it.

**One notable finding:** `lead_time`, which had the strongest *linear correlation* in our EDA (0.293), ranks much lower in actual model importance. That's not a contradiction — once the model already knows the deposit type, it's already captured most of what lead time was indirectly telling it (non-refundable bookings tend to also be made far in advance). This shows the model finds the most *efficient* combination of signals, not just a re-ranking of single-feature correlations."

**If asked — "Isn't relying 60% on one feature risky?"**
"Yes, and it's worth saying plainly: if `deposit_type` data quality or distribution changes at a different hotel or time period, model performance could shift. That's exactly why we'd want to monitor performance in production and retrain regularly — which the fast 1.2-second training time makes easy to do."

---

## 11. Live Prediction Walkthrough (2.5 min)

**Slide bullets — one real booking through the whole pipeline:**
```json
Input:  {"hotel": "Resort Hotel", "lead_time": 350, "adults": 2,
          "deposit_type": "Non Refund", "previous_cancellations": 1,
          "total_of_special_requests": 0}
Output: {"prediction": 1, "label": "Cancelled", "cancellation_probability": 1.0}
```

**Say this:** "Let's walk through exactly what happens to one booking. Say a guest books a Resort Hotel stay 350 days in advance, 2 adults, a non-refundable deposit, has cancelled with us once before, and made zero special requests.

**Step 1 — fill in the gaps.** Real bookings from a partial form or API call won't specify every one of our ~29 features. Any missing field is filled with a sensible default (e.g., a typical lead time or the most common room type), so the model always sees a complete, consistent set of inputs.

**Step 2 — encode.** The categorical fields — hotel type, deposit type, and so on — get converted into the one-hot numeric format the model was trained on, using the *exact same* encoding rules learned from training data (unseen categories are handled safely).

**Step 3 — predict.** The encoded booking goes into the trained XGBoost model, which outputs a raw probability — in this case, effectively **100% probability of cancellation** — driven overwhelmingly by the Non-Refund deposit type, reinforced by the long lead time and prior cancellation history.

**Step 4 — threshold.** Since probability ≥ 50%, we output `prediction: 1`, `label: "Cancelled"`.

**For contrast**, a City Hotel booking made 10 days out, No Deposit, two special requests, and needing a parking space comes back at **0.03% probability** — essentially certain to stay. And a more 'average' booking — City Hotel, 100 days out, No Deposit, standard Online TA channel — comes back at **82% probability of cancellation**, showing the model isn't just a deposit-type on/off switch; lead time and channel still shift the number meaningfully within the No-Deposit population."

---

## 12. Offline vs Online Prediction (2 min)

**Slide bullets:**
| Script | Mode | Use case |
|---|---|---|
| `src/train.py` | — | Trains the model, logs to MLflow, saves to `models/` |
| `src/predict_offline.py` | **Offline / batch** | Score a whole CSV of new bookings at once |
| `src/api.py` | **Online / real-time** | Score one booking instantly via HTTP |

**Say this:** "There are three working parts here. `train.py` is the one place the model actually learns — it's run periodically (e.g., weekly) whenever we want to refresh the model on new data, and it logs everything to MLflow so we can track what changed.

`predict_offline.py` is for **batch** use — say, every morning, we run every booking made in the last 24 hours through the model in one go and get back a CSV with predictions and probabilities added. Good for reporting, dashboards, or a nightly revenue-management review.

`api.py` is for **real-time** use — a live web service a booking system or front-desk application could call the instant a reservation is made, getting an answer back in milliseconds: 'this new booking is 91% likely to cancel.' That's the integration point for something like a live front-desk alert or a dynamic overbooking system."

**If asked — "Why not just always use the API?"**
"Batch processing is far cheaper computationally when you're scoring thousands of bookings at once — no per-request network overhead — and it fits naturally into scheduled reporting jobs. The API exists specifically for the cases where you need an answer *the moment* a booking happens."

---

## 13. MLflow (3 min)

**Slide bullets:**
| Run | max_depth | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time |
|---|---|---|---|---|---|---|---|
| Experiment 1 (baseline) | 6 | 0.882 | 0.861 | 0.813 | 0.836 | 0.954 | 1.2s |
| Experiment 2 (tuned) | 10 | 0.893 | 0.870 | 0.836 | 0.852 | 0.961 | 1.9s |

**Say this:** "Beyond comparing five different model *types*, we also wanted to show a disciplined way of tuning *one* model's settings — this is what MLflow is for.

**Experiment 1 (baseline)** is our current production XGBoost, with its trees allowed to go 6 levels deep. **Experiment 2 (tuned)** is the exact same model, with exactly **one** setting changed: tree depth increased from 6 to 10.

**Why only one parameter changed:** this is a controlled experiment. If we changed five settings at once and performance improved, we wouldn't know *which* change caused it. Changing one variable at a time is the only way to draw a clean conclusion.

**Why max_depth=10 improved performance:** deeper trees can capture more complex combinations of features — remember, our EDA showed cancellation is driven by *interactions* (deposit type combined with lead time combined with channel). A shallower tree can only combine a few features per decision path; a deeper tree can combine more of them, which is exactly the kind of pattern this dataset has.

**Why MLflow, and not just Git:** Git tracks *code* changes — it doesn't track *what happened when we ran that code*: which hyperparameters were used, what accuracy resulted, which exact model file came out the other end. MLflow logs all of that automatically, every time we train, so months from now we can look back and know precisely what configuration produced what result — reproducibility that a code diff alone can't give us.

**Why we have NOT promoted the tuned model to production yet:** even though it scored higher on this one test set, we want to see that improvement hold up on fresh, more recent booking data before switching what's actually serving predictions. Promoting on a single test-set win risks reacting to noise rather than a real, durable improvement. It's a five-minute change to promote it once we're confident — the file already exists at `models/xgboost_tuned_max_depth10.pkl` — but that's a deliberate, considered decision, not an automatic one."

---

## 14. Business Value (2.5 min)

**Say this:** "Bringing this back to what it means for the business:

**Revenue protection.** Every cancellation caught early is a room that can be resold instead of sitting empty. With 81% recall, this model gives advance warning on the large majority of cancellations that would otherwise arrive as a surprise.

**Smarter overbooking.** Hotels already overbook to compensate for expected cancellations — but usually using a flat historical average. A probability *per booking* lets revenue management overbook more precisely, room by room, instead of guessing at a portfolio level.

**Targeted intervention.** High-risk bookings (e.g., 90%+ probability) could trigger a confirmation call, a deposit reminder, or a modified cancellation policy — while low-risk bookings are left alone, saving staff time for where it matters.

**Better forecasting.** Aggregate cancellation-probability data across all upcoming bookings gives a much more accurate occupancy forecast than static historical rates, which helps staffing, procurement, and pricing decisions well beyond just cancellations.

**Decision support, not decision replacement.** This model informs judgment calls — it doesn't automate away the front desk or revenue manager. The output is a probability a human uses, not an automatic cancellation or rebooking action."

---

## 15. Future Improvements (1.5 min)

**Slide bullets:**
- Validate the `deposit_type` pattern on fresh/live data before acting on it commercially
- Promote the tuned model (max_depth=10) after confirming its gain holds on new data
- Probability calibration (make "90%" actually mean 90% in practice)
- Broader hyperparameter search (beyond the single max_depth experiment)
- Monitor for data drift and schedule periodic retraining (cheap — 1.2s)
- Feature engineering: total nights/guests, ADR-per-person, grouped rare countries
- Rebuild SHAP-style explainability for individual predictions ("why did *this* booking score 92%?")
- Pilot with a live hotel booking feed before broad rollout

**Say this:** "Realistic next steps, roughly in priority order: first, validate the deposit-type pattern against live data, since so much of the model's power currently rests on it. Second, confirm the tuned model's improvement holds before promoting it to production. Third, add probability calibration so a '90%' prediction genuinely corresponds to a 90% real-world cancellation rate — important if we want staff to trust and act on specific probability thresholds. And longer-term, richer features, drift monitoring, and a small live pilot before any hotel-wide rollout."

---

## Closing (30 sec)

**Say this:** "To summarize: we built and rigorously compared five models on nearly 120,000 real bookings, selected XGBoost with 88% accuracy and 95% ROC-AUC, deployed it two ways — batch and real-time — and tracked our experimentation formally through MLflow. The next step is validating this in a live setting before it drives real business decisions. Happy to take questions."

---

## 25 Difficult Questions Your Supervisor Might Ask — With Ideal Answers

**1. Why XGBoost over Random Forest — they're almost tied?**
"They're close (95.4% vs 95.3% ROC-AUC), but XGBoost has meaningfully better recall (81.3% vs 76.7%) — it catches more real cancellations, which is the costlier error to miss in this business context. It's also faster to train."

**2. Why not Deep Learning as the primary model?**
"We did train one — the MLP — and it scored 93.9% ROC-AUC, solidly good but below XGBoost's 95.4%, while taking 15x longer to train. Neural networks tend to shine on unstructured data like images or text; for structured, spreadsheet-style tabular data like ours, gradient-boosted trees are usually the stronger, more practical choice."

**3. Why ROC-AUC as the primary metric instead of accuracy?**
"Because our deliverable is a *probability*, not just yes/no. ROC-AUC measures whether the model's probability ranking is trustworthy across every possible threshold, not just the default 50% cutoff. Accuracy alone is also misleading here since 63% of bookings don't cancel — a lazy model could score 63% by doing nothing useful."

**4. Precision vs recall — why prioritize recall?**
"A missed cancellation (false negative) costs the hotel a room that could've been resold. A false alarm (false positive) just costs a wasted follow-up call. The downside of missing real cancellations is bigger, so we favor recall — while still keeping precision high enough (86%) that alerts stay trustworthy."

**5. Could this model be overfitting?**
"We evaluate strictly on a held-out test set of 39,390 bookings the model never saw during training, and we use models like Random Forest and XGBoost that have built-in mechanisms — tree depth limits, subsampling, regularization — specifically to control overfitting. The fact that test-set performance is strong and consistent across five very different model types is further evidence this isn't a fluke of one algorithm."

**6. Does the dataset have bias?**
"Yes, at least one clear one: the extremely strong Non-Refund-deposit-cancels-99%-of-the-time pattern is unusually strong and dataset-specific — we flagged it explicitly rather than presenting it as a universal truth. Any dataset like this also reflects the hotels, time period, and markets it was collected from, so we'd want to validate generalization before wider rollout."

**7. How would you deploy this in production?**
"Two paths already exist: a batch job (`predict_offline.py`) that could run nightly against new bookings, and a real-time API (`api.py`) that a booking system could call the moment a reservation is made. Production deployment would add monitoring, authentication, and a scheduled retraining job on top of what's already built."

**8. Why MLflow instead of just Git?**
"Git tracks code changes; it doesn't track what happened when that code ran — which hyperparameters, what accuracy, which model file resulted. MLflow automatically logs all of that per run, giving us reproducible, comparable experiment history that a code diff alone can't provide."

**9. Why keep the baseline model in production instead of the better-scoring tuned one?**
"The tuned model won on this one test set, but promoting to production on a single win risks reacting to noise rather than a durable improvement. We want that gain to hold up on fresh data before switching what's actually serving live predictions — a deliberate decision, not an automatic one."

**10. What happens if a booking is missing most of its fields?**
"Any field not provided is filled with a sensible default drawn from the training data's typical values, so the model always receives a complete, consistent input — though naturally, more missing information means the prediction leans more heavily on defaults and is less precise for that specific booking."

**11. How do you know the model generalizes beyond this dataset?**
"Honestly, we don't yet — that's the single most important next step (Section 15). Strong test-set performance shows it generalizes *within this dataset's* distribution; validating it on a live, different hotel's data is the real test of broader generalization."

**12. Why one-hot encoding instead of something like target encoding?**
"One-hot is simple, safe against leakage (it doesn't use the target variable to encode features), and works well with tree-based models. We did bucket rare categories to avoid excessive dimensionality, which addresses the usual downside of one-hot encoding on high-cardinality fields."

**13. Why not scale/normalize all the features?**
"Explicit project requirement, and also good practice here: tree-based models (which are 4 of our 5 candidates) are invariant to feature scale, so normalizing them adds complexity with zero benefit. The one exception, the MLP, needs scaled inputs to train properly — so we scale only inside its own internal pipeline, nowhere else."

**14. What's the false positive/false negative cost tradeoff in dollar terms?**
"We haven't quantified that financially yet since we don't have real intervention-cost data — that would be a natural early input for a pilot. What we can say today is the *rate*: 1,920 false alarms vs 2,729 missed cancellations out of 39,390 test bookings, so any financial model should weight the recall side more heavily given it's the larger and typically costlier error type."

**15. How often would this need to be retrained?**
"There's no fixed answer without live monitoring, but practically: training takes about 1.2 seconds, so cost is not the constraint — retraining weekly or even daily against fresh booking data is easy to support. The right cadence would be set once we can monitor for performance drift in production."

**16. Why did you remove `reservation_status`? Isn't more data always better?**
"Not when it's leakage. `reservation_status` records the final outcome — one of its values is literally 'Canceled.' Including it wouldn't make a better model, it would make a *cheating* model that looks perfect in testing but is useless in production, because that field doesn't exist yet at the moment we need to make a prediction."

**17. How confident are you in the feature importance numbers?**
"They come directly from the trained XGBoost model's internal gain-based importance, not an estimate — so they're an accurate reflection of what *this specific model* relies on. Whether that exact weighting holds on a different hotel's data is a separate, open question worth validating."

**18. What if two features contradict each other for a booking?**
"That's exactly the kind of case boosted trees are good at — the model doesn't apply one rule at a time, it weighs all the evidence together through however many trees it takes, arriving at a single balanced probability rather than being confused by a single conflicting signal."

**19. Is the model interpretable enough for staff to trust it?**
"We can extract global feature importance (Section 10) to explain overall model behavior, and the confusion matrix and metrics give a clear accuracy picture. For explaining *individual* predictions ('why did *this* booking score 92%?'), that's a documented future improvement — a SHAP-style explainability layer per booking."

**20. Why 80,000 training records specifically, not the full dataset?**
"That was the target training size from the project brief — roughly two-thirds of the full 119,390-row dataset — leaving 39,390 held out purely for honest testing, which is a solid, statistically meaningful test set size."

**21. What's the risk of the model being wrong in a way that damages guest experience?**
"The main risk is a false positive triggering an unwanted intervention — e.g., a hotel calling a guest who was never going to cancel. That's why we care about precision (86%) alongside recall — it keeps false alarms to a controlled minority (about 1 in 7 flagged bookings) rather than spamming guests."

**22. Could this be gamed — e.g., guests learning to avoid triggering the 'risky' flag?**
"In principle, but the strongest signal (deposit type) is a hotel-controlled policy choice, not something a guest gets to freely set — so gaming risk is limited. It's a fair consideration for any live system that changes guest-facing terms based on a score, and worth reviewing before rollout."

**23. Why not report is_canceled prediction using a different threshold than 50%?**
"50% is the default, but because we output a full probability, the business can choose any threshold that fits its risk tolerance — e.g., only intervene above 70% if false alarms are costly, or above 30% if missing cancellations is far costlier. ROC-AUC being high (0.954) means the ranking stays trustworthy no matter which threshold is chosen."

**24. How do offline and online predictions stay consistent with each other?**
"Both load and call the exact same saved model file and the exact same shared prediction logic in `src/predict.py` — there's no separate implementation for each path, so there's no risk of the two drifting out of sync."

**25. What's the single biggest risk in this project right now?**
"That the model's strongest signal — deposit type — reflects a strong pattern in this particular historical dataset that may not carry over as cleanly to live, current bookings. Before this drives real financial decisions, that assumption needs to be tested against fresh data."

---

## Common Mistakes to Avoid

- **Don't** claim "non-refundable bookings always cancel" as a general truth — it's this dataset's pattern, say so explicitly.
- **Don't** say the model is deployed to production and actively making decisions — it's built, tested, and deployable, but not yet piloted live.
- **Don't** claim a specific dollar ROI figure — we haven't run a financial pilot; only cite the metrics we actually measured.
- **Don't** say "the tuned model is better so we're using it" — the tuned model is *not yet* in production; be precise about that distinction if asked.
- **Don't** conflate correlation (EDA) with model importance (Section 10) — they're related but not identical, and the lead_time discrepancy is worth being ready to explain.
- **Don't** oversell interpretability — we have global feature importance, not yet per-prediction explanations (that's a future item).
- **Don't** rush the confusion matrix — it's the concrete number stakeholders remember most; let it land.

---

## Questions You Should Ask Your Supervisor at the End

1. Is there budget/appetite for a live pilot with real booking data from one property before wider rollout?
2. Who would own the decision to promote the tuned model (max_depth=10) to production — and what evidence threshold would satisfy that?
3. Do we have access to a more recent or different hotel's dataset to test whether the deposit_type pattern generalizes?
4. What's an acceptable false-alarm rate from the business side, to help us pick the right probability threshold?
5. Should the next milestone be the API integration into an existing front-desk/PMS system, or a per-prediction explainability layer (SHAP) first?
6. Is there a data engineering resource to help schedule periodic retraining and monitor for drift once this is live?
