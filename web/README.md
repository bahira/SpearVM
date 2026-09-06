# SpearVM Simulation Lab

Quatre **cas d'usage Three.js prets pour la production** dont la physique est
calculee par les noyaux SIMD AVX2 de SpearVM. Le CPU fait l'algebre lineaire et
les transcendantes vectorisees, le GPU fait la geometrie et les pixels.

```
web/
├── server/          FastAPI + numpy + spur_math  (calcul, WebSocket binaire, REST)
│   ├── spearvm_sim/
│   │   ├── kernels.py     resolution du backend (SpearVM AVX2 ou repli numpy)
│   │   ├── protocol.py    frames binaires spearvm.sim.v1
│   │   ├── bench.py       banc d'essai debit / precision / GFLOPS
│   │   ├── app.py         endpoints REST + /ws/sim/{id} + service du build
│   │   └── sims/          flow_field · wave_field · trainer
│   └── tests/       40 tests (pytest) : noyaux, protocole, simulations, API
└── client/          Vite + TypeScript + Three.js (aucun framework UI)
    └── src/
        ├── core/    viewer partage (1 seul contexte WebGL), types de scene
        ├── net/     decodeur binaire + client WebSocket resilient
        ├── local/   moteurs JS de repli (serveur injoignable)
        └── scenes/  flowField · waveField · trainer · kernelLab
```

## Demarrage rapide

```bash
# 1. noyaux natifs (une fois) — sinon le serveur bascule en numpy de reference
gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC \
    -o spur_math/libspur_kernels.so src/spur_kernels.c -lm

# 2. serveur de calcul (port 8000)
python -m venv .venv && . .venv/bin/activate
pip install -r web/server/requirements.txt
cd web/server && python -m spearvm_sim

# 3. front (port 5173, proxy /api et /ws vers 8000)
cd web/client && npm install && npm run dev
```

Ou, depuis la racine : `make web-install`, `make web-dev`, `make web-test`,
`make web-build` (voir le `Makefile`).

### Production (un seul processus)

```bash
cd web/client && npm run build          # -> web/client/dist
cd ../server  && python -m spearvm_sim  # sert l'API + le build statique
```

ou via Docker :

```bash
docker build -f web/Dockerfile -t spearvm-lab .
docker run --rm -p 8000:8000 spearvm-lab      # CPU AVX2+FMA requis
```

## Les quatre cas d'usage

| # | Scene | Ce que calcule SpearVM | Ce que fait le GPU |
|---|---|---|---|
| 1 | **Champ de flux neuronal** | MLP 2 couches (`matmul_nt_gelu`) sur une grille 3D periodique → potentiel vecteur ; `rot` en differences centrees ; saturation `tanh` | advection ping-pong de 16 k → 262 k particules, echantillonnage trilineaire d'une texture 3D |
| 2 | **Membrane non lineaire** | equation des ondes en differences finies, raidissement `tanh` AVX2 a chaque sous-pas | deplacement du maillage depuis une texture R32F (1 vertex = 1 cellule), eclairage speculaire |
| 3 | **Entrainement live** | forward `matmul_nt` + `gelu`, backward `gelu_backward` + `matmul_backward`, Adam | surface predite coloree par l'erreur + cible filaire, courbe de perte |
| 4 | **Kernel Lab** | `/api/bench` : debit vs numpy, erreur vs IEEE sur [-4,4], GFLOPS matmul, gain de fusion, gradcheck | barres 3D + murs de courbes log |

Les cas 1 a 3 sont pousses en WebSocket ; le cas 4 est un appel REST a la demande.

## Protocole `spearvm.sim.v1`

Chaque frame binaire est auto-descriptif :

```
u32 headerLen | header JSON utf-8 (+ padding 0 jusqu'a un multiple de 4) | payload
```

`header = {kind, tick, time, compute_ms, dtype: "f32"|"i16"|"none", shape, scale, stats}`.
Les champs volumineux (vitesses, hauteurs) partent en **int16 quantifie**
(`valeur = i16 * scale`) : bande passante divisee par deux, erreur bornee a
1 LSB. Ordres de grandeur mesures : flux 81 ko a 12 Hz (~1 Mo/s), membrane
50 ko a 30 Hz (~1,5 Mo/s), entrainement 21 ko a 20 Hz.

Messages client → serveur (texte JSON) : `{type:"params", params:{…}}`,
`{type:"rate", value:Hz}`, `{type:"command", …}`, `{type:"ping", t}`.
Le **schema des parametres est fourni par le serveur** (`/api/simulations`) et
l'UI lil-gui est generee a partir de lui : ajouter un parametre a une
simulation Python suffit, aucun code client a toucher.

## Endpoints

| Methode | Route | Role |
|---|---|---|
| GET | `/api/health` | etat, backend de noyaux, clients connectes |
| GET | `/api/simulations` | catalogue + schema des parametres |
| GET | `/api/bench?quick=true` | banc d'essai (cache 30 s) |
| GET | `/api/gradcheck` | verification differences finies de `gelu_backward` |
| WS  | `/ws/sim/{id}` | flux de frames binaires |
| GET | `/*` | build client (SPA) si present |

## Robustesse

- **Degradation en cascade** : noyaux AVX2 → repli numpy exact (`SPEARVM_FORCE_FALLBACK=1`
  pour le forcer) → si le serveur est injoignable, le client bascule sur un
  moteur JavaScript local a resolution reduite. Le bandeau indique toujours le
  mode reel ; aucune mesure n'est presentee comme native si elle ne l'est pas.
- **Cadence maitrisee** : boucle a horloge fixe cote serveur, resynchronisation
  si retard, calcul dans un thread (`asyncio.to_thread`) pour ne jamais bloquer
  la boucle d'evenements, reconnexion exponentielle cote client.
- **Limites** : `SPEARVM_MAX_CLIENTS` (defaut 8), cadence bornee
  `SPEARVM_MIN_RATE`/`SPEARVM_MAX_RATE`, une instance de simulation par
  connexion (aucun etat partage).
- **Un seul contexte WebGL** partage par les scenes, dispose explicite des
  geometries/materiaux/render targets au changement d'onglet.

## Variables d'environnement

| Variable | Defaut | Role |
|---|---|---|
| `SPEARVM_HOST` / `SPEARVM_PORT` | `0.0.0.0` / `8000` | ecoute |
| `SPEARVM_STATIC` | `web/client/dist` | build servi |
| `SPEARVM_MAX_CLIENTS` | `8` | connexions simultanees |
| `SPEARVM_MAX_RATE` | `60` | cadence max (Hz) |
| `SPEARVM_CORS` | `*` | origines autorisees |
| `SPEARVM_LOG_LEVEL` | `info` | verbosite uvicorn/app |
| `SPEARVM_FORCE_FALLBACK` | — | force le backend numpy |

## Note de mesure (trouvee en instrumentant ce lab)

Le serveur alterne appels **BLAS** (numpy) et regions **OpenMP** (noyaux
SpearVM). Par defaut OpenBLAS laisse ses threads *spinner* ~20 ms apres chaque
appel : sur une machine 2 vCPU, la region OpenMP suivante est affamee et un
`matmul_nt` 256³ passe de **0.4 ms a 18 ms (×45)**. Le package positionne donc
`OPENBLAS_THREAD_TIMEOUT=1` (et `OMP_WAIT_POLICY=ACTIVE`) *avant* l'import de
numpy — surchargeables par l'environnement. Le banc d'essai mesure en plus par
blocs alternes (min de 3 iterations consecutives par variante) pour que les
deux implementations voient les memes conditions.

## Tests

```bash
cd web/server && python -m pytest tests -q     # noyaux, protocole, sims, API
cd web/client && npm run build                 # tsc --noEmit + build vite
```

Les tests serveur verifient notamment : le **datasheet de precision** des
noyaux (gelu 0.079 / quintique 0.0174 / erf 2e-5), l'equivalence
`matmul_nt_gelu` ≡ `gelu(matmul_nt)` sur des tailles non alignees, le
gradcheck par differences finies, la **convergence reelle** de l'entrainement
(perte divisee par 5 en 60 ticks), la stabilite du schema d'ondes et
l'incompressibilite du champ de flux.
