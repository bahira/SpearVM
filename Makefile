# Makefile SpearVM (baseline AVX2, pas de -march=native)
CC      = gcc
CFLAGS  = -O3 -mavx2 -mfma
JITCFLAGS = -O2 -mavx2 -mfma -Iinclude -fPIC
LDLIBS  = -lm
# JIT : src/spur.c = Win64 (windows.h) ; src/spur_posix.c = Linux/macOS (System V)
ifeq ($(OS),Windows_NT)
JITSRC = src/spur.c
JITLIBS =
JITLIB = bin/spur.dll
else
JITSRC = src/spur_posix.c
JITLIBS = -lpthread
JITLIB = bin/spur.so
endif

.PHONY: all kernels jit encoding bench clean test

all: kernels jit encoding bench

kernels: bin/spur_kernels.dll
jit: $(JITLIB)

bin/spur_kernels.dll: src/spur_kernels.c
	@mkdir -p bin
	$(CC) $(CFLAGS) -fopenmp -shared -o $@ src/spur_kernels.c $(LDLIBS)

$(JITLIB): $(JITSRC) include/spur.h
	@mkdir -p bin
	$(CC) $(JITCFLAGS) -shared -o $@ $(JITSRC) $(LDLIBS) $(JITLIBS)

encoding: bin/test_encoding.exe

bin/test_encoding.exe: tests/test_encoding.c
	@mkdir -p bin
	$(CC) -O2 -mavx2 -static -o $@ tests/test_encoding.c

bench: bin/bench_native_vs_jit.exe

ifeq ($(OS),Windows_NT)
BENCHSRC = examples/bench_native_vs_jit.c
else
BENCHSRC = examples/bench_native_vs_jit_posix.c
endif

bin/bench_native_vs_jit.exe: $(BENCHSRC) $(JITLIB)
	@mkdir -p bin
	$(CC) -O3 -Iinclude $(BENCHSRC) $(JITLIB) -o $@ $(LDLIBS)

test: all
	./bin/test_encoding.exe
	PYTHONPATH=. python -m pytest tests/ -q

clean:
	rm -rf bin dist build *.egg-info spur_math/_native \
	       spur_math/spur_kernels.dll spur_math/spur_jit.dll \
	       nn_io.bin
