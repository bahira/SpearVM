# Makefile SpearVM (baseline AVX2, pas de -march=native)
CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma
JITCFLAGS = -O2 -mavx2 -mfma -Iinclude -fPIC
LDLIBS  = -lm

.PHONY: all kernels jit encoding bench clean test

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

clean:
	rm -rf bin dist build *.egg-info spur_math/_native \
	       spur_math/spur_kernels.dll spur_math/spur_jit.dll \
	       map_dbg.bin nn_io.bin
