# Plan de relance des expériences

Rédigé le 2026-10-12 sur la branche `refactor-pipeline`. Il s'adresse à la personne, ou à la session Claude, qui reprend le travail sur la machine avec accès HPC. Il donne, dans l'ordre, tout ce qu'il faut faire pour regénérer **chaque** chiffre, tableau et figure du papier « Do Adaptive Scorer Mixtures Improve Token Selection? A Controlled Study in GPT-2 » (soumis à ARR en août 2026, à resoumettre).

Noms des méthodes, papier → configuration :

| Papier | Configuration |
|---|---|
| S_E-only | `se_only` |
| S_U / S_L / S_D only | `su_only` / `sl_only` / `sd_only` |
| Mix-4 | `mix4` |
| Mix-3 | `mix3` |
| VN-Mix | `vn_mix` |
| Mix-4 + bigram | `mix4_bigram` |
| Mix-4 + noise | `mix4_noise` |
| CLM complet | `clm` |

Les anciens noms TS-4, TS-3 et VN-TS correspondent à Mix-4, Mix-3 et VN-Mix.

## 0. Contexte et décisions

**Bugs du code des anciens runs.** Aucun résultat du papier actuel n'est réutilisable.

| # | Bug |
|---|---|
| K1 | Masque calculé tous les 50 pas puis réappliqué à d'autres batchs : la sélection par score ne portait que sur 2 % des pas, pour toutes les méthodes. |
| K14 | S_U et S_L décalés d'une position. |
| K16 | S_U valait −Σp², et non l'entropie. |
| T16 | Récompense créditée à des poids tirés au pas même de l'évaluation. |
| K15 | Probablement aucune perte de référence pour S_E (voir §9). |

**Décisions des auteurs (2026-10-11)**
1. **Protocole** : on continue l'entraînement de GPT-2 pré-entraîné (`gpt2`, `gpt2-large`), avec un batch de 8 (Small) ou 4 (Large) séquences de 1024 tokens, sur 25 000 pas. Le texte de l'annexe B sera corrigé (« from scratch », batch 512).
2. **Couverture** : grille complète, seeds 42, 43 et 44.
3. **Le reste** : code aligné sur le papier (§3, Algorithm 1, annexe B).

**Branches**
- `fix-per-step-mask` : le correctif, puis l'alignement sur le papier (commit `d965f77`). C'est l'implémentation de référence.
- `refactor-pipeline` (celle-ci) : la même méthode dans une architecture modulaire, pilotée par YAML (voir `README.md`).

## 1. État du code : PRÊT

- **Tests** : `python -m pytest -q tests` (22 tests, CPU, environ 40 s) couvre les scores, les mélanges, la sélection, la perte, le schedule, la configuration, les dossiers de run et des entraînements complets avec un petit modèle.
- **Non-régression** : `python tests/regression/compare_reference.py --reference <checkout de fix-per-step-mask>`. Sur un petit GPT-2, les 7 méthodes donnent pas à pas les mêmes masques, pertes, poids tirés, récompenses et perplexités que l'implémentation de référence.
- **Différences voulues avec la référence**, toutes deux alignées sur le papier :
  - nombre de tokens retenus = ⌈α(T−1)⌉ (le ⌈αB⌉ du papier, sur les T−1 positions qui ont une cible), au lieu de round(αT). Pour T = 1024, c'est identique à α = 0,3 / 0,5 / 0,7 (307 / 512 / 717), et 921 au lieu de 922 à α = 0,9 ;
  - la graine du bruit de `mix4_noise` vaut par défaut celle du run (`selection.noise_seed: null`), au lieu d'une graine fixe 12345.

Rappel de la méthode, telle qu'implémentée :
- S_E = ℓθ − ℓref, avec la référence figée (`selection.reference_model: gpt2`) exécutée sur le même batch ;
- S_U = entropie ; S_L = −(logit₁ − logit₂) ; S_D = −log fréquence du corpus ;
- z-norm sur les positions 1..T−1 ; top-k par séquence à chaque pas ;
- poids w ~ Dir(α) tirés tous les 100 pas ; après chaque bloc de 1000 pas, r = max(0, L_prev − L_val) et α ← α + r·w (VN-Mix : r·w/(σ̂ + 1e-3), avec σ̂ = 1 tant qu'il y a moins de 2 récompenses) ;
- perte = moyenne sur les tokens sélectionnés ;
- AdamW (3e-4, β = (0,9 ; 0,95), weight decay 0,1), warmup de 500 pas, puis cosinus jusqu'à 10 % ;
- poids fp32, autocast bf16 sur CUDA ;
- évaluations intermédiaires sur 50 batchs, évaluation finale sur tout le jeu de validation.

**Si GPT-2 Large manque de mémoire** : activer le gradient checkpointing dans `sieve/models/factory.py`. **Ne jamais réduire le batch** : cela changerait le protocole.

## 2. Installation

```bash
git clone https://github.com/Tankiit/MODE.git
cd MODE
git checkout refactor-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests
# Les nœuds de calcul sont souvent hors ligne : tout télécharger avant
huggingface-cli download gpt2
huggingface-cli download gpt2-large
mkdir -p data logs
```

## 3. Préparation des données (une seule fois, avec internet)

```bash
python scripts/prepare_data.py --dataset wikitext103 --out ./data/wikitext103 --bigram
python scripts/prepare_data.py --dataset openwebmath --out ./data/openwebmath_1b \
    --train_tokens 1000000000 --val_tokens 1000000 --bigram
```

| Fichier | Contenu |
|---|---|
| `train.bin`, `val.bin` | Tokens `uint16`, tokenizer GPT-2. WikiText-103 : HF `Salesforce/wikitext` (`wikitext-103-raw-v1`), splits `train` et `validation`. OpenWebMath : HF `open-web-math/open-web-math` en streaming ; 1 000 000 000 tokens d'entraînement, puis 1 000 000 de validation, avec `<\|endoftext\|>` après chaque document et sans document commun aux deux. |
| `freq.pt` | S_D(v) = −log((count(v) + 1) / (N + V)) |
| `bigram_freq.pt` | log p(tok \| prev) par maximum de vraisemblance. Fallback pour les paires jamais vues : la plus petite valeur observée moins log 2. |
| `meta.json` | Sources, nombres de tokens et choix de construction |

Le script se termine par `os._exit(0)`, car le streaming de `datasets` peut bloquer la sortie de Python : c'est voulu.

## 4. Vérifications avant la grille

**1. Grille de fumée sur CPU.** Elle prend environ 15 minutes et parcourt toute la chaîne : entraînement, dossiers de run et les trois analyses. Les commandes sont en tête de `configs/sweeps/smoke.yaml`.

**2. Fumée sur GPU et pilote de durée et de mémoire** pour GPT-2 Large :

```bash
for m in se_only su_only sl_only sd_only mix4 mix3 vn_mix mix4_bigram mix4_noise clm; do
  python main.py --config configs/base.yaml --config configs/models/gpt2.yaml \
    --config configs/data/wikitext103.yaml --config configs/methods/$m.yaml \
    --set training.steps=300 --set eval.interval=100 --set output.root=runs_gpu_smoke
done
python main.py --config configs/base.yaml --config configs/models/gpt2-large.yaml \
  --config configs/data/owm.yaml --config configs/methods/mix4.yaml \
  --set training.steps=1000 --set output.root=runs_pilot
```

À vérifier :
- `history.json` contient `n_scored_steps` égal au nombre de pas (0 pour `clm`) ;
- `sample_log` contient les pas 0, 100 et 200 ;
- aucune valeur NaN ;
- `peak_memory_gb` et `wall_time_s` du pilote.

Prendre la durée du pilote × 25, avec de la marge, comme `--time` des runs Large.

## 5. Grille de runs : 183 runs

```bash
python scripts/expand_sweep.py configs/sweeps/paper.yaml --out sweeps/paper
# sweeps/paper/list_gpt2.txt : 75 runs ; sweeps/paper/list_gpt2-large.txt : 108 runs
```

| Groupe | Modèle | Contenu | Runs |
|---|---|---|---|
| B | GPT-2 Small | S_E-only et Mix-4 × {WikiText-103, OWM} × α ∈ {0,3 ; 0,5 ; 0,7 ; 0,9} × 3 seeds, 25k pas (annexe C) | 48 |
| C | GPT-2 Small | Contrôle à nombre de tokens retenus égal (environ 71,7M, promis au reviewer remz), WikiText-103 : S_E-only et Mix-4 avec α = 0,3 / 0,5 / 0,7 / 0,9 sur 29 167 / 17 500 / 12 500 / 9 722 pas ; CLM complet sur 8 750 pas ; 3 seeds | 27 |
| A1 | GPT-2 Large | Tableau 1 : les 9 méthodes à α = 0,7 × 2 corpus × 3 seeds | 54 |
| A2 | GPT-2 Large | Tableau 2 : S_E-only, Mix-4 et VN-Mix × α ∈ {0,3 ; 0,5 ; 0,9} × 2 corpus × 3 seeds | 54 |

Avec le seed 42, `output.save_final` est activé pour 6 runs A1 : S_E-only, S_U, S_L, S_D et Mix-4 sur WikiText-103, plus Mix-4 sur OpenWebMath. Chaque fichier de `sweeps/paper/` est une configuration complète, lisible et autonome.

## 6. Lancement sur le HPC

```bash
mkdir -p logs
sbatch -A <compte> --time=03:00:00 --array=1-75%20 \
    --export=ALL,LIST=sweeps/paper/list_gpt2.txt scripts/run_array.slurm
sbatch -A <compte> --time=<hh:mm:ss> --array=1-108%20 \
    --export=ALL,LIST=sweeps/paper/list_gpt2-large.txt scripts/run_array.slurm
```

Adapter `module load` et l'environnement en tête de `scripts/run_array.slurm`.
- Chaque run crée un nouveau dossier `runs/<nom>_<date-heure>_<id>/` et n'écrase jamais rien.
- `--skip-if-done` saute une configuration déjà terminée (même empreinte de configuration et présence de `DONE`). On peut donc relancer le même array après des échecs.
- Les dossiers sans `DONE` sont des runs interrompus : les analyses les ignorent.

## 7. Analyse : tableaux et figures du papier

```bash
python -m sieve.analysis.aggregate --runs runs --out results
python -m sieve.analysis.bandit_plots --runs runs --out figs
python -m sieve.analysis.selection_overlap --runs runs --out results     # GPU recommandé
```

| Élément du papier | Source | Sortie |
|---|---|---|
| Tableau 1 (`tab:main`) : moyenne ± écart-type (ddof = 1) | A1 | `results/table1.md` |
| Tableau 2 (`tab:alpha-sweep`) | A2, plus les cellules α = 0,7 de A1 | `results/table2.md` |
| Écarts appariés (`tab:paired-gaps`) : Mix-4 − S_E-only par seed | A1, A2, B | `results/paired.md` |
| Contrôle à tokens égaux : tableau **séparé** | C | `results/matched.md` |
| Figure 1 (`fig:bandit-dynamics`) | `mix4_bigram` et `mix4_noise`, WikiText-103, seeds 42 à 44 | `figs/fig_seed_dominance_bandit_dynamics.png` |
| Figure 2 (`fig:arm-trajectories`) | mêmes runs | `figs/fig_arm_trajectories_grid.pdf` |
| Chiffres de l'annexe « Bandit Dynamics » | mêmes runs | `figs/bandit_numbers.json` |
| Spearman (`tab:correlations`) | `final.pt` de Mix-4, seed 42, chaque corpus | `results/correlations.md` |
| Jaccard (`tab:jaccard`) | `final.pt` WikiText-103, seed 42 | `results/jaccard.md` |
| Annexe B, « Compute » | `wall_time_s` | `results/compute.md` |

Les analyses ne lisent que les runs terminés. Si une configuration a été terminée plusieurs fois, elles prennent le run le plus récent.

## 8. Après les runs : texte du papier à mettre à jour

- **Annexe B.**
  - Continuation d'entraînement depuis le checkpoint public.
  - Batch de 4 ou 8 × 1024, soit 102,4M (Large) ou 204,8M (Small) tokens traités par run.
  - Poids fp32 et activations bf16 ; matériel et heures-GPU (`results/compute.md`).
  - Taille du jeu de validation.
- **S_E.** La référence est calculée sur le même batch, avec le même contexte.
- **S_D et bigramme.** Donner le lissage et le fallback.
- **Retenue.** ⌈α(T−1)⌉ tokens par séquence.
- **Variante noise.** Le bruit est tiré à chaque pas, avec la graine du run.
- **VN-Mix.** σ̂ = 1 tant qu'il y a moins de 2 récompenses.
- **Tout chiffre cité** (abstract, résultats, conclusion, annexes) doit être recalculé. **Les conclusions peuvent changer.** Les rapporter telles qu'elles sortent.

## 9. Points ouverts

1. **À décider avant les groupes B et C : la référence de S_E pour GPT-2 Small.** La cible part de `gpt2`, qui est aussi la référence. S_E vaut donc 0 au départ, puis mesure l'écart au point de départ ; ce n'est pas le même signal que pour Large. Il faut choisir une autre référence (`--set selection.reference_model=distilgpt2`, ou une ligne dans `configs/models/gpt2.yaml`), ou assumer ce choix et le décrire.
2. **K15, à confirmer avec le co-auteur.** Ni `modal_prepare_dataset.py` ni `jeanzay/setup_env.sh` (branche `ARR_March_2026`) ne calculent de pertes de référence : dans les anciens runs, S_E était probablement la NLL brute. Utile seulement pour décrire les anciens runs.
