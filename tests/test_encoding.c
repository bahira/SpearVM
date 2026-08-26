/* test_encoding.c — validation octet par octet des encodages x64 du map kernel
   Win64 ABI. Chaque helper d'emission est compare a la sequence de reference.
   Puis test d'execution reelle : code genere applique out[i]=in0[i]+in1[i].
   Build: gcc -O2 tests/test_encoding.c -o bin/test_encoding.exe
   Run  : bin/test_encoding.exe  -> exit 0 si tout passe                    */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

static uint8_t buf[512];
static int pos;

static void reset(void) { pos = 0; }
static void e8(uint8_t b) { buf[pos++] = b; }
static void e32(uint32_t v) { memcpy(buf + pos, &v, 4); pos += 4; }

/* ---------- helpers d'emission (a porter tels quels dans spur.c) ------- */

/* movsd xr_dst, [r12]  : F2 41 0F 10 04 24   (SIB obligatoire: r12&7=100) */
static void ld0(int xd) {
    e8(0xF2); e8(xd >= 8 ? 0x42 : 0x41); e8(0x0F); e8(0x10);
    e8((uint8_t)(0x04 | ((xd & 7) << 3))); /* mod00 reg=xd rm=SIB */
    e8(0x24);                              /* scale0 idx=none base=r12(B=1) */
}

/* movsd xr_dst, [r13+0] : F2 41 0F 10 4D 00 (mod01 disp8: r13&7=101 serait
   RIP-relatif en mod00)                                                   */
static void ld1(int xd) {
    e8(0xF2); e8(xd >= 8 ? 0x42 : 0x41); e8(0x0F); e8(0x10);
    e8((uint8_t)(0x44 | ((xd & 7) << 3) | 0x01)); /* mod01 reg=xd rm=101 */
    e8(0x00);
}

/* movsd [r14], xr_src : F2 41 0F 11 04 26   (SIB base=110+r14, PAS 0x24!)
   0x24 serait base=r12 -> ecraserait in0 silencieusement                   */
static void st0(int xs) {
    e8(0xF2); e8(xs >= 8 ? 0x42 : 0x41); e8(0x0F); e8(0x11);
    e8((uint8_t)(0x04 | ((xs & 7) << 3)));
    e8(0x26);
}

/* addsd xmm0,xmm1 : F2 0F 58 C1 ; op=58 add 59 mul 5C sub 5E div 5F max 5D min */
static void sd_op(uint8_t opc) { e8(0xF2); e8(0x0F); e8(opc); e8(0xC1); }

/* movsd xr, [rip+disp32] : F2 0F 10 05 d32 (constantes pool) */
static void ld_rip(int xr, int32_t disp) {
    e8(0xF2); e8(xr >= 8 ? 0x42 : 0x40); e8(0x0F); e8(0x10);
    e8((uint8_t)(0x05 | ((xr & 7) << 3))); /* mod00 reg=xr rm=101(RIP) */
    e32((uint32_t)disp);
}

/* add r64,imm8 : 49 83 C{reg} 08  (r12..r14) */
static void add_r_imm8(int r) {
    e8(0x49); e8(0x83); e8((uint8_t)(0xC0 | (r & 7))); e8(0x08);
}

/* dec r15 : 49 FF CF ; jne rel32 : 0F 85 xx xx xx xx */
static void dec_jnz(int32_t rel) { e8(0x49); e8(0xFF); e8(0xCF); e8(0x0F); e8(0x85); e32((uint32_t)rel); }

/* ---------- harness de comparaison -------------------------------------- */
static int fails;
static void expect(const char* name, const uint8_t* exp, int n) {
    if (pos != n || memcmp(buf, exp, (size_t)n) != 0) {
        fails++;
        printf("FAIL %-22s attendu", name);
        for (int i = 0; i < n; i++) printf(" %02X", exp[i]);
        printf("\n%25s emis   ", "");
        for (int i = 0; i < pos; i++) printf(" %02X", buf[i]);
        printf("\n");
    } else {
        printf("PASS %s\n", name);
    }
}

/* ---------- test d'execution : out[i]=in0[i]+in1[i] ---------------------- */
typedef void (*mapfn)(const double*, const double*, double*, long long);

static int run_smoke(void) {
    SYSTEM_INFO si; GetSystemInfo(&si);
    DWORD sz = (DWORD)((si.dwPageSize + 511) / 512 * 512);
    uint8_t* code = VirtualAlloc(NULL, sz, MEM_COMMIT | MEM_RESERVE,
                                 PAGE_EXECUTE_READWRITE);
    if (!code) return printf("FAIL smoke: VirtualAlloc\n"), 1;
    printf("  [smoke] alloc OK %p\n", (void*)code);

    /* prologue: r12=rcx(in0) r13=rdx(in1) r14=r8(out) r15=r9(n) */
    int p = 0;
    code[p++] = 0x49; code[p++] = 0x89; code[p++] = 0xCC;          /* mov r12,rcx (rex49!) */
    code[p++] = 0x49; code[p++] = 0x89; code[p++] = 0xD5;          /* mov r13,rdx */
    code[p++] = 0x4D; code[p++] = 0x89; code[p++] = 0xC6;          /* mov r14,r8  */
    code[p++] = 0x4D; code[p++] = 0x89; code[p++] = 0xCF;          /* mov r15,r9  */
    int loop_start = p; /* cible du back-jump */

    /* corps: xmm0=[r12]; xmm1=[r13]; addsd xmm0,xmm1; [r14]=xmm0 */
    code[p++] = 0xF2; code[p++] = 0x41; code[p++] = 0x0F; code[p++] = 0x10;
    code[p++] = 0x04; code[p++] = 0x24;                            /* xmm0=[r12] */
    code[p++] = 0xF2; code[p++] = 0x41; code[p++] = 0x0F; code[p++] = 0x10;
    code[p++] = 0x4D; code[p++] = 0x00;                            /* xmm1=[r13] */
    code[p++] = 0xF2; code[p++] = 0x0F; code[p++] = 0x58; code[p++] = 0xC1; /* add */
    code[p++] = 0xF2; code[p++] = 0x41; code[p++] = 0x0F; code[p++] = 0x11;
    code[p++] = 0x04; code[p++] = 0x26;                            /* [r14]=xmm0 */

    /* queue: add r12/r13/r14,8; dec r15; jnz corps(-38) */
    int tail_start = p;
    code[p++] = 0x49; code[p++] = 0x83; code[p++] = 0xC4; code[p++] = 0x08;
    code[p++] = 0x49; code[p++] = 0x83; code[p++] = 0xC5; code[p++] = 0x08;
    code[p++] = 0x49; code[p++] = 0x83; code[p++] = 0xC6; code[p++] = 0x08;
    code[p++] = 0x49; code[p++] = 0xFF; code[p++] = 0xCF;          /* dec r15 */
    int rel = -(p + 6 - loop_start);                                /* vers corps */
    code[p++] = 0x0F; code[p++] = 0x85;
    memcpy(code + p, &rel, 4); p += 4;                             /* jnz */
    code[p++] = 0xC3;                                              /* ret */

    FlushInstructionCache(GetCurrentProcess(), code, p);
    printf("  [smoke] code ecrit (%d octets), appel...\n", p);

    double a[4] = {1.5, -2.0, 3.25, 0.0};
    double b[4] = {0.5, 2.0, -0.25, 7.0};
    double o[4] = {0, 0, 0, 0};
    ((mapfn)code)(a, b, o, 4);
    printf("  [smoke] retour d'appel\n");

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        double expct = a[i] + b[i];
        if (o[i] != expct) { ok = 0; printf("FAIL smoke [%d]: %f != %f\n", i, o[i], expct); }
    }
    if (ok) printf("PASS execution smoke (out[i]=in0[i]+in1[i], 4 elems)\n");
    VirtualFree(code, 0, MEM_RELEASE);
    return !ok;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0); /* survit au crash */
    /* --- encodages unitaires --- */
    reset(); ld0(0);
    { uint8_t e[] = {0xF2,0x41,0x0F,0x10,0x04,0x24}; expect("ld0 xmm0,[r12]", e, 6); }
    reset(); ld0(9);
    { uint8_t e[] = {0xF2,0x42,0x0F,0x10,0x0C,0x24}; expect("ld0 xmm9,[r12]", e, 6); }
    reset(); ld1(1);
    { uint8_t e[] = {0xF2,0x41,0x0F,0x10,0x4D,0x00}; expect("ld1 xmm1,[r13]", e, 6); }
    reset(); st0(0);
    { uint8_t e[] = {0xF2,0x41,0x0F,0x11,0x04,0x26}; expect("st0 [r14],xmm0", e, 6); }
    reset(); sd_op(0x58);
    { uint8_t e[] = {0xF2,0x0F,0x58,0xC1}; expect("addsd x0,x1", e, 4); }
    reset(); sd_op(0x59);
    { uint8_t e[] = {0xF2,0x0F,0x59,0xC1}; expect("mulsd x0,x1", e, 4); }
    reset(); sd_op(0x5C);
    { uint8_t e[] = {0xF2,0x0F,0x5C,0xC1}; expect("subsd x0,x1", e, 4); }
    reset(); sd_op(0x5E);
    { uint8_t e[] = {0xF2,0x0F,0x5E,0xC1}; expect("divsd x0,x1", e, 4); }
    reset(); sd_op(0x5F);
    { uint8_t e[] = {0xF2,0x0F,0x5F,0xC1}; expect("maxsd x0,x1", e, 4); }
    reset(); sd_op(0x5D);
    { uint8_t e[] = {0xF2,0x0F,0x5D,0xC1}; expect("minsd x0,x1", e, 4); }
    reset(); ld_rip(3, -42);
    { uint8_t e[] = {0xF2,0x40,0x0F,0x10,0x1D,0xD6,0xFF,0xFF,0xFF};
      expect("ld_rip xmm3,-42", e, 9); }
    reset(); add_r_imm8(12);
    { uint8_t e[] = {0x49,0x83,0xC4,0x08}; expect("add r12,8", e, 4); }
    reset(); add_r_imm8(14);
    { uint8_t e[] = {0x49,0x83,0xC6,0x08}; expect("add r14,8", e, 4); }

    /* --- prologue : movs reg,reg (piege REX.R si src>=8 seulement) --- */
    reset();
    e8(0x49); e8(0x89); e8(0xCC); /* mov r12,rcx : src=rcx(<8)->R=0 */
    { uint8_t e[] = {0x49,0x89,0xCC}; expect("mov r12,rcx", e, 3); }
    reset();
    e8(0x49); e8(0x89); e8(0xD5); /* mov r13,rdx */
    { uint8_t e[] = {0x49,0x89,0xD5}; expect("mov r13,rdx", e, 3); }
    reset();
    e8(0x4D); e8(0x89); e8(0xC6); /* mov r14,r8 : src>=8 -> R=1 */
    { uint8_t e[] = {0x4D,0x89,0xC6}; expect("mov r14,r8", e, 3); }
    reset();
    e8(0x4D); e8(0x89); e8(0xCF); /* mov r15,r9 */
    { uint8_t e[] = {0x4D,0x89,0xCF}; expect("mov r15,r9", e, 3); }

    printf("\n");
    fails += run_smoke();

    printf("\n=== %s (%d echec(s)) ===\n", fails ? "ECHEC" : "TOUS PASS", fails);
    return fails;
}
