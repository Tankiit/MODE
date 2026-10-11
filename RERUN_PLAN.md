# Plan de relance des expériences SIEVE

Rédigé le 2026-10-11 sur la branche `fix-per-step-mask`. Il s'adresse à la personne, ou à la session Claude, qui reprend le travail sur la machine avec accès HPC. Il donne, dans l'ordre, tout ce qu'il faut faire pour regénérer **chaque** chiffre, tableau et figure du papier « Do Adaptive Scorer Mixtures Improve Token Selection? A Controlled Study in GPT-2 » (soumis à ARR en août 2026, à resoumettre).

Noms des variantes : S_E-only (excess loss seule), S_U / S_L / S_D only, Mix-4, Mix-3 (sans S_E), VN-Mix, Mix-4 + bigram, Mix-4 + noise. Anciens noms : TS-4, TS-3, VN-TS.

## 0. Contexte et décisions

**Bugs trouvés dans le code des anciens runs.** Aucun résultat du papier actuel n'est réutilisable.

| # | Bug | Effet |
|---|---|---|
| K1 | Le masque était calculé tous les 50 pas, puis réappliqué à de nouveaux batchs tirés au hasard. | La sélection par score ne portait que sur 2 % des pas, pour toutes les méthodes, S_E-only compris. |
| K14 | S_U et S_L étaient décalés d'une position par rapport au token sélectionné. | — |
| K16 | `score_entropy` calculait −Σp², et non l'entropie : `exp_()` en place écrasait `log_p`. | S_U n'a jamais été l'entropie. |
| T16 | La récompense allait à des poids tirés au début du pas d'évaluation lui-même. | Ces poids n'avaient servi qu'à une seule mise à jour. |
| K15 | Probablement aucune perte de référence pour S_E (voir §9). | — |

**Décisions des auteurs (2026-10-11)**
1. **Protocole** : celui qui a réellement tourné. On **continue l'entraînement** de GPT-2 pré-entraîné (`gpt2` et `gpt2-large` sur Hugging Face), avec des batchs de 8 séquences (Small) ou 4 séquences (Large) de 1024 tokens, sur 25 000 pas. C'est le texte de l'annexe B qui sera corrigé (« from scratch », batch 512, 13B tokens), pas le code.
2. **Couverture** : grille complète, seeds 42, 43 et 44 partout.
3. **Tout le reste** : le code est aligné sur le papier (§3, Algorithm 1, annexe B).

**Références dans le papier** : §3 (Eq. `utility`, `reward`, `update`, `train_loss`) et annexe B (hyperparamètres, variantes, VN-Mix Eq. `vnts`).

## 1. Code aligné sur le papier : FAIT

Testé sur CPU : `python -m pytest -q tests` (10 tests), plus des runs de bout en bout sur de petits corpus avec `--model nanogpt`.

| Point | Ce que fait maintenant le code | Fichiers |
|---|---|---|
| Masque | Scores et masque sont calculés à chaque pas, sur le batch courant. | `state.py`, `rho1.py`, `mask.py`, `loop.py` |
| T1 (K14) | Pour S_U et S_L, la colonne t décrit la distribution qui prédit le token t (`logits[t-1]`), comme pour S_E et S_D. | `scorers.py` |
| K16 | S_U = vraie entropie −Σ p log p. | `scorers.py` |
| T2 | S_L = −(logit₁ − logit₂). | `scorers.py` |
| T3 | S_E = ℓθ − ℓref, sans clamp. Erreur si la référence manque. S_D lève aussi une erreur sans `freq.pt`. | `scorers.py` |
| Référence de S_E | GPT-2 Small figé (`--ref_model gpt2`), **exécuté sur le batch courant**, donc avec le même contexte que le modèle entraîné. Pas de table pré-calculée : sur des fenêtres fixes, la référence verrait un autre contexte que les séquences tirées à des positions aléatoires. Seules les variantes qui utilisent S_E chargent la référence. | `reference.py`, `loop.py`, `train_sieve.py` |
| z-norm | Statistiques calculées sur les positions 1..T−1, celles qui ont un terme de perte. | `scorers.py` |
| T4 à T6 | Un seul Dirichlet ; α ← α + r·w ; r = max(0, L_prev − L_val), avec L_val = NLL moyenne de validation. La première évaluation sert seulement de point de départ. | `bandit.py`, `state.py` |
| T7 | VN-Mix : α ← α + r·w / (σ̂ + 1e-3), où σ̂ est l'écart-type de r·w sur toutes les évaluations passées. σ̂ = 1 tant qu'il y a moins de 2 récompenses (choix à documenter dans le papier). | `bandit.py` |
| T16 | L'évaluation a lieu après chaque bloc de 1000 pas complets, avant le tirage suivant. La récompense crédite donc le mélange actif sur les 100 derniers pas. | `loop.py` |
| T8 | T_rescore = 100, T_val = 1000 (valeurs par défaut). | `config.py` |
| T9 | AdamW(3e-4, β = (0,9 ; 0,95), weight decay 0,1), warmup linéaire de 500 pas, puis décroissance cosinus jusqu'à 10 % à `max_steps`. Clip à 1,0. | `loop.py`, `config.py` |
| T10 | Poids en fp32, forward sous autocast bf16 sur CUDA (validation aussi). | `train_sieve.py`, `loop.py` |
| T11 | Perte = moyenne sur les tokens sélectionnés, sur tous les chemins. | `loss.py`, `loop.py` |
| T12 | Options `--out_dir`, `--log_interval`, `--save_final` ; `--batch_size` et `--seq_len` servent aux tests seulement. Voir le contenu de `history.json` ci-dessous. | `train_sieve.py`, `loop.py`, `state.py` |
| T13 | Évaluations intermédiaires sur 50 batchs de 8 × 1024 tokens ; évaluation finale (le chiffre rapporté) sur **tout** le jeu de validation. | `loop.py` |
| T14 | `scripts/prepare_data.py` (voir §3). | |
| T15 | `scripts/aggregate.py`, `scripts/plot_bandit.py`, `scripts/analyze_selection.py` (voir §7). | |

`history.json` contient :
- les arguments et la config ;
- la perte, le lr et E[w] à chaque log ;
- `val_ppl` et `eval_steps` ;
- `sample_log` (poids tirés) et `reward_log` ;
- `alpha_final` et `expected_weights_final` ;
- `final_val_ppl` ;
- le timing et `wall_time_s`.

Il est écrit en dernier et de façon atomique. `final.pt` est le state_dict du modèle.

**Si GPT-2 Large manque de mémoire** avec les poids en fp32 et les logits complets : activer `model.gradient_checkpointing_enable()`. **Ne jamais réduire le batch** : cela changerait le protocole.

## 2. Installation

```bash
git clone https://github.com/Tankiit/MODE.git
cd MODE
git checkout fix-per-step-mask
python -m venv .venv && source .venv/bin/activate
pip install -e ".[liger]" scipy matplotlib pytest
python -m pytest -q tests            # 10 tests, CPU, environ 10 s
# Les nœuds de calcul sont souvent hors ligne : tout télécharger avant
huggingface-cli download gpt2
huggingface-cli download gpt2-large
mkdir -p data runs logs
```

## 3. Préparation des données

Ces commandes sont à lancer une seule fois, sur une machine avec internet :

```bash
python scripts/prepare_data.py --dataset wikitext103 --out ./data/wikitext103 --bigram
python scripts/prepare_data.py --dataset openwebmath --out ./data/openwebmath_1b \
    --train_tokens 1000000000 --val_tokens 1000000 --bigram
```

Les répertoires correspondent à `DATASET_DIRS` dans `train_sieve.py` : `--dataset wikitext103` lit `./data/wikitext103`, et `--dataset owm` lit `./data/openwebmath_1b`.

| Fichier | Contenu |
|---|---|
| `train.bin`, `val.bin` | Tokens `uint16`, tokenizer GPT-2. WikiText-103 : HF `Salesforce/wikitext` (`wikitext-103-raw-v1`), splits `train` et `validation`, lignes concaténées comme dans le texte d'origine. OpenWebMath : HF `open-web-math/open-web-math` en streaming ; les 1 000 000 000 premiers tokens vont dans `train.bin`, les 1 000 000 suivants dans `val.bin`, avec `<\|endoftext\|>` après chaque document et sans document à cheval sur les deux. |
| `freq.pt` | float32 [50257] : S_D(v) = −log((count(v) + 1) / (N + V)) |
| `bigram_freq.pt` | Pour Mix-4 + bigram : log p(tok \| prev) par maximum de vraisemblance sur `train.bin`. Fallback pour les paires jamais vues : la plus petite valeur observée moins log 2. |
| `meta.json` | Sources, nombres de tokens et choix de construction |

Vérifiez les nombres de tokens dans les logs et dans `meta.json`. Le script se termine par `os._exit(0)`, car le streaming de `datasets` peut bloquer la sortie de Python : c'est voulu.

## 4. Tests de fumée sur GPU, avant la grille

```bash
for v in "rho1" "su_only" "sl_only" "sd_only" "sieve" "sieve_no_se" \
         "sieve --variance_normalized_bandit" "sieve --sd_scorer bigram_freq" \
         "sieve --sd_scorer random_noise" "clm"; do
  tag=$(echo "$v" | tr ' -' '__')
  python train_sieve.py --model gpt2 --dataset wikitext103 --baseline $v --alpha 0.7 \
      --steps 300 --eval_interval 100 --log_interval 100 --seed 42 --out_dir runs/smoke_$tag
done
# Pilote de durée et de mémoire, GPT-2 Large :
python train_sieve.py --model gpt2l --dataset owm --baseline sieve --alpha 0.7 \
    --steps 1000 --eval_interval 1000 --log_interval 100 --seed 42 --out_dir runs/pilot_large
```

À vérifier pour chaque run :
- le log affiche `Scored steps: 300/300 (100.0% of steps)` (sauf pour `clm`) ;
- `history.json` existe ;
- `sample_log` contient les pas 0, 100 et 200 ;
- aucune valeur NaN.

Prendre la durée du pilote × 25, avec de la marge, comme `--time` des runs Large.

## 5. Grille de runs : 183 runs

```bash
bash scripts/make_commands.sh
# commands_small.txt: 75 runs, commands_large.txt: 108 runs
```

| Groupe | Modèle | Contenu | Runs |
|---|---|---|---|
| B | GPT-2 Small | S_E-only et Mix-4 × {WikiText-103, OWM} × α ∈ {0,3 ; 0,5 ; 0,7 ; 0,9} × 3 seeds, 25k pas (annexe C) | 48 |
| C | GPT-2 Small | Contrôle à nombre de tokens retenus égal (environ 71,7M, promis au reviewer remz), sur WikiText-103 : S_E-only et Mix-4 avec α = 0,3 sur 29 167 pas, α = 0,5 sur 17 500, α = 0,7 sur 12 500, α = 0,9 sur 9 722 ; CLM complet sur 8 750 pas ; 3 seeds | 27 |
| A1 | GPT-2 Large | Tableau 1 : les 9 variantes à α = 0,7 × 2 corpus × 3 seeds | 54 |
| A2 | GPT-2 Large | Tableau 2 : S_E-only, Mix-4 et VN-Mix × α ∈ {0,3 ; 0,5 ; 0,9} × 2 corpus × 3 seeds (α = 0,7 est déjà dans A1) | 54 |

| Variante | Flags |
|---|---|
| S_E-only | `--baseline rho1` |
| S_U / S_L / S_D only | `--baseline su_only` / `sl_only` / `sd_only` |
| Mix-4 | `--baseline sieve` |
| Mix-3 | `--baseline sieve_no_se` |
| VN-Mix | `--baseline sieve --variance_normalized_bandit` |
| Mix-4 + bigram | `--baseline sieve --sd_scorer bigram_freq` |
| Mix-4 + noise | `--baseline sieve --sd_scorer random_noise` |

`--save_final` est ajouté sur 6 runs A1 du seed 42 : S_E-only, S_U, S_L, S_D et Mix-4 sur WikiText-103, plus Mix-4 sur OpenWebMath.

Optionnel : une référence CLM sans sélection à 25k pas (`--baseline clm`) par modèle et par corpus, utile pour discuter l'effet de α.

## 6. Lancement sur le HPC

```bash
mkdir -p logs runs
# 1) GPT-2 Small d'abord : plus rapide, et permet de valider les résultats avant Large
sbatch -A <compte> --time=03:00:00 --array=1-75%20 \
    --export=ALL,CMDS=commands_small.txt scripts/run_array.slurm
# 2) GPT-2 Large, avec --time déduit du pilote (§4)
sbatch -A <compte> --time=<hh:mm:ss> --array=1-108%20 \
    --export=ALL,CMDS=commands_large.txt scripts/run_array.slurm
```

Adapter `module load` et l'environnement en tête de `scripts/run_array.slurm`. Les runs qui ont déjà un `history.json` sont sautés : on peut relancer le même array après des échecs.

Volumes : un run Large traite 25 000 × 4 × 1024 = 102,4M tokens, un run Small (batch 8) en traite 204,8M.

## 7. Analyse : produire les tableaux et figures du papier

```bash
python scripts/aggregate.py --runs runs --out results            # Tableaux 1 et 2, écarts appariés, contrôle C, compute
python scripts/plot_bandit.py --runs runs --out figs             # Figures 1 et 2, chiffres de l'annexe Bandit Dynamics
python scripts/analyze_selection.py --runs runs --out results    # Tableaux Jaccard et Spearman (GPU recommandé)
```

| Élément du papier | Source | Sortie |
|---|---|---|
| Tableau 1 (`tab:main`) : moyenne ± écart-type (ddof = 1) sur 3 seeds | A1 | `results/table1.md` |
| Tableau 2 (`tab:alpha-sweep`) | A2, plus les cellules α = 0,7 de A1 | `results/table2.md` |
| Écarts appariés (`tab:paired-gaps`) : Mix-4 − S_E-only, signé, par seed | A1, A2, B | `results/paired.md` |
| Contrôle à tokens égaux : tableau **séparé**, avec ses réglages | C | `results/matched.md` |
| Figure 1 (`fig:bandit-dynamics`) | Mix-4 + bigram et Mix-4 + noise, WikiText-103, seeds 42 à 44 | `figs/fig_seed_dominance_bandit_dynamics.png` |
| Figure 2 (`fig:arm-trajectories`) | mêmes runs | `figs/fig_arm_trajectories_grid.pdf` |
| Chiffres de l'annexe « Bandit Dynamics » (L1 même seed et entre seeds, entropies terminales) | mêmes runs | `figs/bandit_numbers.json` |
| Spearman (`tab:correlations`) : 1 000 tokens de validation | `final.pt` de Mix-4, seed 42, chaque corpus | `results/correlations.md` |
| Jaccard (`tab:jaccard`) : même batch de validation ; Mix-4 avec E[w] final | `final.pt` WikiText-103, seed 42 | `results/jaccard.md` |
| Annexe B, « Compute » | `wall_time_s` des `history.json` | `results/compute.md` |

## 8. Après les runs : texte du papier à mettre à jour

- **Annexe B.**
  - Remplacer « from scratch » par une continuation d'entraînement depuis le checkpoint public.
  - Corriger le batch (4 ou 8 × 1024) et le nombre de tokens par run.
  - Mettre à jour la précision (poids fp32, activations bf16), le matériel et les heures-GPU.
  - Préciser la taille du jeu de validation : évaluations intermédiaires sur 409 600 tokens, évaluation finale sur tout le jeu.
- **S_E.** Indiquer que la perte de référence est calculée sur le même batch, avec le même contexte.
- **S_D et bigramme.** Donner la formule de lissage de S_D, l'estimation par maximum de vraisemblance et le fallback du bigramme.
- **Variante noise.** Le bruit est retiré à chaque pas (à chaque scoring), et non « à chaque rescoring ».
- **VN-Mix.** Indiquer σ̂ = 1 tant qu'il y a moins de 2 récompenses.
- **Tout chiffre cité dans le texte** (abstract, résultats, conclusion, annexes : 0,04 PPL, Jaccard, etc.) doit être recalculé à partir des nouveaux runs. **Les conclusions peuvent changer.** Les rapporter telles qu'elles sortent.

## 9. Points ouverts

1. **À décider avant les groupes B et C : la référence de S_E pour GPT-2 Small.** Avec le protocole retenu, la cible GPT-2 Small part du checkpoint `gpt2`, qui est aussi la référence (`--ref_model gpt2`). S_E vaut donc 0 au départ, puis mesure seulement l'écart entre le modèle entraîné et son point de départ. Ce n'est pas le même signal que pour GPT-2 Large (cible Large, référence Small). Il faut choisir une autre référence pour Small (par exemple `distilgpt2`), ou assumer ce choix et le décrire.
2. **K15, à confirmer avec le co-auteur.** Ni `modal_prepare_dataset.py` ni `jeanzay/setup_env.sh` (branche `ARR_March_2026`) ne calculent de pertes de référence. Dans les anciens runs, S_E était donc probablement la NLL brute du modèle entraîné. Cela ne change rien à la relance, mais il faut le savoir pour décrire correctement les anciens runs.
3. **Bruit de Mix-4 + noise.** Le générateur a une graine fixe (12345, `scorers.py`), indépendante de `--seed`. Les trois seeds voient donc la même suite de bruit. Ce n'est pas gênant pour la comparaison bigramme / bruit à seed égal, mais il faut le savoir.
