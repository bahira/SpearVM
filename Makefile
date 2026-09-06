# Makefile SpearVM (baseline AVX2, pas de -march=native)
CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma
JITCFLAGS = -O2 -mavx2 -mfma -Iinclude -fPIC
LDLIBS  = -lm

.PHONY: all kernels jit encoding bench clean test \
        web-install web-dev web-serve web-build web-test web-docker

all: kernels jit encoding bench

kernels: bin/spur_kernels.dll
jit: bin/spur.dll

bin/spur_kernels.dll: src/spur_kernels.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -fopenmp -shared -o $@ src/spur_kernels.c $(LDLIBS)

bin/spur.dll: src/spur.c include/spur.h
	@mkdir -p bin
	$(CC) $(JITCFLAGS) -shared -o $@ src/spur.c $(LDLIBS)

encoding: bin/test_encoding.exe

bin/test_encoding.exe: tests/test_encoding.c
	@mkdir -p bin
	$(CC) -O2 -mavx2 -static -o $@ tests/test_encoding.c

bench: bin/bench_native_vs_jit.exe

bin/bench_native_vs_jit.exe: examples/bench_native_vs_jit.c bin/spur.dll
	@mkdir -p bin
	$(CC) -O3 -Iinclude examples/bench_native_vs_jit.c bin/spur.dll -o $@ $(LDLIBS)

test: all
	./bin/test_encoding.exe
	PYTHONPATH=. python -m pytest tests/ -q

# ---------------------------------------------------------------------------
# SpearVM Simulation Lab (web/) — voir web/README.md
# ---------------------------------------------------------------------------
PY      ?= python3
VENV    ?= .venv
VPY      = $(VENV)/bin/python

$(VENV)/bin/activate:
	$(PY) -m venv $(VENV)
	$(VPY) -m pip install -U pip
	$(VPY) -m pip install -r web/server/requirements-dev.txt

web-install: $(VENV)/bin/activate spur_math/libspur_kernels.so
	cd web/client && npm install

spur_math/libspur_kernels.so: src/spur_kernels.c
	$(CC) $(CFLAGS) -fopenmp -shared -fPIC -o $@ $< $(LDLIBS)

# serveur de calcul seul (API + WebSocket sur :8000)
web-serve: $(VENV)/bin/activate spur_math/libspur_kernels.so
	cd web/server && ../../$(VPY) -m spearvm_sim

# front en dev (proxy /api et /ws vers :8000) — lancer web-serve a cote
web-dev:
	cd web/client && npm run dev

web-build:
	cd web/client && npm run build

web-test: $(VENV)/bin/activate spur_math/libspur_kernels.so
	cd web/server && ../../$(VPY) -m pytest tests
	cd web/client && npm run typecheck

web-docker:
	docker build -f web/Dockerfile -t spearvm-lab .

clean:
	rm -rf bin dist build *.egg-info spur_math/_native \
	       spur_math/spur_kernels.dll spur_math/spur_jit.dll \
	       web/client/dist map_dbg.bin nn_io.bin
