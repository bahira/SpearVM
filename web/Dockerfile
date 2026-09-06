# SpearVM Simulation Lab — image unique (client build + serveur de calcul)
# Contexte de build attendu : racine du depot.
#   docker build -f web/Dockerfile -t spearvm-lab .
#   docker run --rm -p 8000:8000 spearvm-lab
# CPU requis : x86-64 avec AVX2 + FMA (sinon repli numpy automatique).

FROM node:22-slim AS client
WORKDIR /build
COPY web/client/package*.json ./
RUN npm ci
COPY web/client/ ./
RUN npm run build


FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SPEARVM_HOST=0.0.0.0 \
    SPEARVM_PORT=8000 \
    SPEARVM_STATIC=/app/web/client/dist \
    OMP_WAIT_POLICY=PASSIVE

# gcc + libgomp : compilation des noyaux AVX2 au build (pas au premier import)
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/server/requirements.txt web/server/requirements.txt
RUN pip install --no-cache-dir -r web/server/requirements.txt

COPY src/ src/
COPY include/ include/
COPY spur_math/ spur_math/
COPY web/server/ web/server/
COPY scripts/ scripts/
COPY --from=client /build/dist web/client/dist

RUN gcc -O3 -mavx2 -mfma -fopenmp -shared -fPIC \
        -o spur_math/libspur_kernels.so src/spur_kernels.c -lm \
 && apt-get purge -y gcc && apt-get autoremove -y

RUN useradd --create-home --uid 10001 spearvm && chown -R spearvm /app
USER spearvm

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/ready').status==200 else 1)"

WORKDIR /app/web/server
CMD ["python", "-m", "spearvm_sim"]
