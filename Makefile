# Makefile SpearVM
CC      = gcc
CFLAGS  = -O3 -Wall -Iinclude
LDLIBS  = -lm

all: bin/spur.dll bin/test_spur.exe

bin/spur.dll: src/spur.c include/spur.h
	@mkdir -p bin
	$(CC) $(CFLAGS) -shared -o $@ src/spur.c $(LDLIBS)

bin/test_spur.exe: examples/test_spur.c bin/spur.dll
	@mkdir -p bin
	$(CC) $(CFLAGS) -Iinclude examples/test_spur.c -Lbin -lspur -o $@ $(LDLIBS)

bench: bin/bench_native_vs_jit.exe
	cd bin && ./bench_native_vs_jit.exe

bin/bench_native_vs_jit.exe: examples/bench_native_vs_jit.c
	@mkdir -p bin
	$(CC) -O3 examples/bench_native_vs_jit.c -o $@ $(LDLIBS)

clean:
	rm -rf bin jit_dll.bin

.PHONY: all bench clean
