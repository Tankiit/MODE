# Plan de relance des expériences SIES_E

Rédigé le 2026-10-11 sur la branche `fix-per-step-mask`. Il s'adresse à la personne, ou à la session Claude, qui reprend le travail sur la machine avec accès HPC. Il donne, dans l'ordre, tout ce qu'il faut faire pour regénérer **chaque** chiffre, tableau et figure du papier « Do Adaptive Scorer Mixtures Improve Token Selection? A Controlled Study in GPT-2 » (soumis à ARR en août 2026, à resoumettre).

## 0. Contexte et décisions

- **Bug corrigé** (commit `9ed891e`). Dans les anciens runs, le masque de sélection était calculé tous les 50 pas puis réappliqué à de nouveaux batchs tirés au hasard. La sélection par score ne portait donc que sur 2 % des pas, pour toutes les méthodes, \seonly compris. **Aucun résultat du papier actuel n'est réutilisable.**
- **Décisions des auteurs (2026-10-11)**
  1. **Protocole** : celui qui a réellement tourné. On **continue l'entraînement** de GPT-2 pré-entraîné (`gpt2` et `gpt2-large` sur Hugging Face), avec des batchs de 8 séquences (Small) ou 4 séquences (Large) de 1024 tokens, sur 25 000 pas. C'est le texte de l'annexe B qui sera corrigé (« from scratch », batch 512, 13B tokens), pas le code.
  2. **Couverture** : grille complète, seeds 42, 43 et 44 partout.
  3. **Tout le reste** : le code est aligné sur le papier (§3, Algorithm 1, annexe B), voir §1.
- **Références dans le papier** : §3 (Eq. `utility`, `reward`, `update`, `train_loss`) et annexe B (hyperparamètres, variantes, VN-Mix Eq. `vnts`). Les noms de variantes du papier sont Mix-4, Mix-3 et VN-Mix (anciennement TS-4, TS-3 et VN-TS).

## 1. Modifications de code à faire AVANT tout run

Faites-les sur cette branche, en un commit par point, avec un test unitaire CS_U pour chacun.

| # | Fichier | Code actuel | Cible (papier) |
|---|---|---|---|
| T1 | `sieve/scorers.py` (`score_entropy`, `score_logit_spread`) | Le score à la position t est calculé sur `logits[t]`, qui prédit le token t+1, alors que `mask[t]` sélectionne le token t (prédit par `logits[t-1]`). Le score est donc décalé d'une position. | Calculer sur `logits[:, :-1]` puis `F.pad(..., (1, 0))`, comme `score_excess_loss`. \S_D et le bigramme sont déjà alignés. |
| T2 | `sieve/scorers.py` (`score_logit_spread`) | `1 / (top1 - top10 + 1e-6)` | `-(top1 - top2)` : marge logit négative (§3) |
| T3 | `sieve/scorers.py` (`score_excess_loss`) | `clamp(min=0)` ; sans `ref_losses`, retombe en silence sur la NS_L brute | `ℓθ − ℓref` sans clamp. **Lever une erreur** si `ref_losses` est absent. |
| T4 | `sieve/bandit.py`, `sieve/state.py` | Bandit contextuel : 16 bins, dont 4 actifs | Un seul Dirichlet (`n_bins = 1`, `encode_context` renvoie 0) |
| T5 | `sieve/bandit.py` (`update`) | `α ← 0.95·α + 2·r·w` si r > 0, sinon `+ 0.1` | `α ← α + r·w` (Eq. `update`) : ni discount, ni plancher |
| T6 | `sieve/state.py` (`_ema_reward`) | `r = �S_L / EMA(\|�S_L\|)`, signé | `r = max(0, L_prev − L_val)` (Eq. `reward`), avec `L_val` = NS_L moyenne de validation, et `r = 0` à la première évaluation |
| T7 | `sieve/bandit.py` (VN-Mix) | `α += 2·r·w·σ`, σ sur une fenêtre de 10 avec clip | `α ← α + r·w / (σ̂ + ε)`, ε = 1e-3, où σ̂_k est l'écart-type courant de `r_j·w_j^(k)` sur toutes les évaluations passées (Eq. `vnts`). Choisir σ̂ tant qu'il y a moins de 2 observations, et le documenter. |
| T8 | `sieve/config.py` | `rescore_interval = 50` | `100` (T_rescore) |
| T9 | `sieve/loop.py` | `AdamW(lr, weight_decay=0.1)`, β₂ = 0,999, lr constant | `AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)`, warmup linéaire sur 500 pas puis décroissance cosinus jusqu'à 10 % du pic à `max_steps`. Clip à 1,0 (déjà le cas). |
| T10 | `train_sieve.py` | Modèle chargé entièrement en bf16 | Poids en fp32, forward sous autocast bf16 (déjà fourni par `DeviceCfg.autocast`) |
| T11 | `sieve/loss.py` | `selective_loss` divise par B(T−1) (chemin CS_U) | Moyenne sur les tokens sélectionnés (Eq. `train_loss`), comme le chemin CUDA de `loop.py` |
| T12 | `train_sieve.py`, `sieve/loop.py` | Rien n'est sauvegardé | Ajouter trois flags : `--out_dir`, qui écrit `history.json` (voir le détail sous le tableau) ; `--log_interval` ; `--save_final`, qui écrit `final.pt` (state_dict) |
| T13 | `sieve/loop.py` | L'évaluation finale ne porte que sur 50 batchs | Évaluation finale sur **tout** le jeu de validation, en gardant 50 batchs pour les évaluations intermédiaires |
| T14 | nouveau `scripts/prepare_data.py` | `prepare_sieve.py` ne gère ni WikiText-103, ni les bigrammes, ni les pertes de référence à grande échelle | Voir §3 |
| T15 | nouveaux `scripts/aggregate.py`, `scripts/analyze_selection.py`, `scripts/plot_bandit.py` | — | Voir §7 |

Contenu de `history.json` (T12) :
- la config ;
- `val_ppl` à chaque évaluation, avec le pas ;
- les poids espérés E[w] à chaque log ;
- les poids tirés à chaque rescoring ;
- les récompenses ;
- le timing ;
- la perplexité finale.

**Tests unitaires à ajouter** (`tests/`, sur CS_U) :
- alignement : `SS_U[t]` doit être l'entropie de la distribution qui prédit le token t ;
- le masque est recalculé à chaque pas, et les poids sont retirés tous les 100 pas ;
- les règles de mise à jour T5 à T7 reproduisent les équations sur un exemple chiffré ;
- la perte est normalisée par le nombre de tokens sélectionnés.

Un test de fumée équivalent au premier existe déjà : il a servi à valider `9ed891e` avec un petit modèle et des données synthétiques.

**Si GPT-2 Large manque de mémoire** avec les poids en fp32 : activer `model.gradient_checkpointing_enable()`. **Ne jamais réduire le batch** : cela changerait le protocole.

## 2. Installation

```bash
git clone https://github.com/Tankiit/MOS_E.git
cd MOS_E
git checkout fix-per-step-mask
python -m venv .venv && source .venv/bin/activate
pip install -e ".[liger]" scipy
# Télécharger modèles et tokenizer (les nœuds de calcul sont souvent hors ligne)
huggingface-cli download gpt2
huggingface-cli download gpt2-large
mkdir -p data runs logs
```

## 3. Préparation des données (`scripts/prepare_data.py`, T14)

Commandes, à lancer une seule fois :

```bash
python scripts/prepare_data.py --dataset wikitext103 --out ./data/wikitext103 --ref_model gpt2 --bigram
python scripts/prepare_data.py --dataset openwebmath --out ./data/openwebmath_1b \
    --train_tokens 1000000000 --val_tokens 1000000 --ref_model gpt2 --bigram
```

Les répertoires correspondent à `DATASET_DIRS` dans `train_sieve.py` : `wikitext103` vers `./data/wikitext103`, et `owm` vers `./data/openwebmath_1b`.

Le script doit écrire les fichiers suivants dans chaque répertoire :

| Fichier | Contenu |
|---|---|
| `train.bin`, `val.bin` | Tokens `uint16` (tokenizer GPT-2), avec `<\|endoftext\|>` (50256) après chaque document. WikiText-103 : HF `wikitext` / `wikitext-103-raw-v1`, splits `train` et `validation`. OpenWebMath : HF `open-web-math/open-web-math` en streaming (split `train`) ; les 1 000 000 000 premiers tokens vont dans `train.bin`, les 1 000 000 suivants dans `val.bin`, sans recouvrement. |
| `meta.json` | Source, tokenizer, nombres de tokens, choix de lissage |
| `freq.pt` | float32 [50257] : `−log f(v)`, avec `f(v) = (count(v) + 1) / (N + V)` (\S_D, §3) |
| `ref_losses.bin` | float32, une valeur par token de `train.bin` : NS_L du token i sous `gpt2` (GPT-2 Small figé, annexe B), calculée sur des fenêtres consécutives de 1024 tokens, avec 0 en position 0 de chaque fenêtre. C'est le format lu par `sieve/data.py` (`ref_losses[o:o+T]`). Environ 4 Go pour OpenWebMath. |
| `bigram_freq.pt` | dict `{keys: int64 triés (prev*V + tok), log_freq: float32 log p(tok\|prev), vocab_size: 50257, fallback_log_freq: float}`, au format lu par `TokenScorers.score_bigram_freq` (\SS_D^bigram = −log p(y_t \| y_{t−1}) sur tout le corpus d'entraînement, annexe B). Le lissage et le fallback sont à choisir, puis à noter dans `meta.json`. Pour OpenWebMath, compter les paires par morceaux (1B tokens). |

Vérifications :
- les nombres de tokens ;
- `len(ref_losses.bin) == len(train.bin)` ;
- la NS_L moyenne de référence est plausible (environ 3 nats sur WikiText-103) ;
- `freq.pt.shape == (50257,)`.

## 4. Tests de fumée sur GS_U (avant la grille)

```bash
for v in "rho1" "su_only" "sl_only" "sd_only" "sieve" "sieve_no_se" \
         "sieve --variance_normalized_bandit" "sieve --sd_scorer bigram_freq" \
         "sieve --sd_scorer random_noise"; do
  tag=$(echo "$v" | tr ' -' '__')
  python train_sieve.py --model gpt2 --dataset wikitext103 --baseline $v --alpha 0.7 \
      --steps 300 --eval_interval 100 --log_interval 100 --seed 42 --out_dir runs/smoke_$tag
done
# Pilote de durée, GPT-2 Large :
python train_sieve.py --model gpt2l --dataset owm --baseline sieve --alpha 0.7 \
    --steps 1000 --eval_interval 1000 --log_interval 100 --seed 42 --out_dir runs/pilot_large
```

À vérifier pour chaque run :
- le log affiche `Scored steps: 300/300 (100.0% of steps)` ;
- `history.json` existe ;
- les poids sont tirés aux pas 0, 100 et 200 ;
- aucune valeur NaN.

Une fois le pilote terminé, multiplier sa durée par 25, ajouter de la marge, et prendre ce résultat comme `--time` du job Slurm pour les runs Large.

## 5. Grille de runs : 183 runs

```bash
bash scripts/make_commands.sh
# commands_small.txt: 75 runs, commands_large.txt: 108 runs
```

| Groupe | Modèle | Contenu | Runs |
|---|---|---|---|
| B | GPT-2 Small | \seonly et Mix-4 × {WikiText-103, OWM} × α ∈ {0,3 ; 0,5 ; 0,7 ; 0,9} × 3 seeds, 25k pas (annexe C) | 48 |
| C | GPT-2 Small | Contrôle à nombre de tokens retenus égal (environ 71,7M, promis au reviewer remz), sur WikiText-103 : \seonly et Mix-4 avec α = 0,3 sur 29 167 pas, α = 0,5 sur 17 500, α = 0,7 sur 12 500, α = 0,9 sur 9 722 ; CLM complet sur 8 750 pas ; 3 seeds | 27 |
| A1 | GPT-2 Large | Tableau 1 : les 9 variantes à α = 0,7 × 2 corpus × 3 seeds | 54 |
| A2 | GPT-2 Large | Tableau 2 : \seonly, Mix-4 et VN-Mix × α ∈ {0,3 ; 0,5 ; 0,9} × 2 corpus × 3 seeds (α = 0,7 est déjà dans A1) | 54 |

Correspondance entre variantes et flags (définie dans `make_commands.sh`) :

| Variante | Flags |
|---|---|
| \seonly | `rho1` |
| \S_U / \S_L / \S_D seul | `su_only` / `sl_only` / `sd_only` |
| Mix-4 | `sieve` |
| Mix-3 | `sieve_no_se` |
| VN-Mix | `sieve --variance_normalized_bandit` |
| Mix-4 + bigram | `sieve --sd_scorer bigram_freq` |
| Mix-4 + noise | `sieve --sd_scorer random_noise` |

`--save_final` est ajouté sur 6 runs A1 du seed 42 : \seonly, \S_U, \S_L, \S_D et Mix-4 sur WikiText-103, plus Mix-4 sur OpenWebMath.

Optionnel : une référence CLM sans sélection à 25k pas (`--baseline clm`) pour chaque modèle et corpus, utile pour discuter l'effet de α.

## 6. Lancement sur le HPC

```bash
mkdir -p logs runs
# 1) GPT-2 Small d'abord : plus rapide, et permet de valider les résultats avant de lancer Large
sbatch -A <compte> --time=03:00:00 --array=1-75%20 \
    --export=AS_L,CMDS=commands_small.txt scripts/run_array.slurm
# 2) GPT-2 Large, avec --time déduit du pilote (§4)
sbatch -A <compte> --time=<hh:mm:ss> --array=1-108%20 \
    --export=AS_L,CMDS=commands_large.txt scripts/run_array.slurm
```

Adapter `module load` et l'environnement en tête de `scripts/run_array.slurm`. Les runs déjà terminés, c'est-à-dire ceux qui ont un `history.json`, sont sautés : on peut relancer le même array après des échecs.

Ordres de grandeur, à confirmer par le pilote : un run Large traite 25 000 × 4 × 1024 = 102,4M tokens, et un run Small (batch 8) en traite 204,8M.

## 7. Sorties à produire et éléments correspondants du papier

| Élément du papier | Source | Script |
|---|---|---|
| Tableau 1 (`tab:main`) : moyenne ± écart-type sur 3 seeds | A1 | `aggregate.py` |
| Tableau 2 (`tab:alpha-sweep`) | A2, plus les cellules α = 0,7 de A1 | `aggregate.py` |
| Tableau des écarts appariés (`tab:paired-gaps`) : Mix-4 − \seonly, signé, par seed | A1, A2, B | `aggregate.py` |
| Contrôle à tokens égaux : tableau **séparé**, avec ses réglages | C | `aggregate.py` |
| Figure 1 (`fig:bandit-dynamics`) : distances L1 entre poids espérés, et E[w_S_D] terminal | `history.json` de Mix-4 + bigram et Mix-4 + noise, WikiText-103, seeds 42 à 44 | `plot_bandit.py` |
| Figure 2 (`fig:arm-trajectories`) : trajectoires des 4 poids | mêmes runs | `plot_bandit.py` |
| Chiffres de l'annexe « Bandit Dynamics » : L1 même seed et entre seeds, entropies terminales | mêmes runs | `plot_bandit.py` |
| Corrélations de Spearman (`tab:correlations`) entre les 4 scorers, sur 1 000 tokens de validation au pas 25k | `final.pt` de Mix-4, seed 42, sur chaque corpus | `analyze_selection.py` |
| Jaccard (`tab:jaccard`) entre les masques de \seonly, \S_U, \S_L, \S_D et Mix-4, sur un même batch de validation | `final.pt` WikiText-103 du seed 42 ; Mix-4 avec ses poids espérés finaux | `analyze_selection.py` |
| Annexe B, « Compute » : durée par run, GS_U, heures-GS_U totales | `history.json` (timing) et logs Slurm | `aggregate.py` |

## 8. Après les runs : texte du papier à mettre à jour

- **Annexe B.** Remplacer « from scratch » par une continuation d'entraînement depuis le checkpoint public. Corriger le batch (4 ou 8 × 1024) et le nombre de tokens par run. Mettre à jour la précision (fp32 + bf16), le matériel et les heures-GS_U. Préciser la taille du jeu de validation.
- **Variante noise.** Le bruit est retiré à chaque pas (chaque scoring), et non « à chaque rescoring ».
- **VN-Mix.** Documenter la valeur de σ̂ au démarrage (T7).
- **Tout chiffre cité dans le texte** (abstract, résultats, conclusion, annexes : 0,04 PS_L, Jaccard, etc.) doit être recalculé à partir des nouveaux runs. **Les conclusions peuvent changer.** Les rapporter telles qu'elles sortent.

## 9. Point ouvert à confirmer avec le co-auteur (K15)

Ni `modal_prepare_dataset.py` ni `jeanzay/setup_env.sh` (branche `ARR_March_2026`) ne calculent `ref_losses.bin`. Dans les anciens runs, \S_E était donc probablement la NS_L brute du modèle entraîné, sans référence. Cela ne change rien à la relance, puisque T3 et T14 imposent la référence, mais c'est à confirmer pour décrire correctement les anciens runs.
