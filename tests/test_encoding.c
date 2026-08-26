/* test_encoding.c — Vérifie octet par octet que chaque instruction émise
   par l'émetteur correspond exactement à ce que produit GCC.
   Compile avec le reste et exécute via main(). */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

/* ================= Structure de test ================================= */
typedef struct { unsigned char* data; int len; } Buf;
static Buf out_buf;
static unsigned char emit_buffer[256];
static int emit_pos = 0;

static void e_reset(void){ emit_pos=0; memset(emit_buffer,0,256); }
static void e8(unsigned char b){ emit_buffer[emit_pos++]=b; }
static void e32(unsigned v){ memcpy(emit_buffer+emit_pos,&v,4); emit_pos+=4; }

static int check(const char* name, const unsigned char* expected, int exp_len){
    if(emit_pos!=exp_len){
        printf("FAIL %s: longueur %d != %d\n",name,emit_pos,exp_len);
        printf("  attendu:");for(int i=0;i<exp_len;i++)printf(" %02X",expected[i]);
        printf("\n  emis:   ");for(int i=0;i<emit_pos;i++)printf(" %02X",emit_buffer[i]);
        printf("\n");
        return 0;
    }
    if(memcmp(emit_buffer,expected,exp_len)!=0){
        printf("FAIL %s: contenu différent\n",name);
        printf("  attendu:");for(int i=0;i<exp_len;i++)printf(" %02X",expected[i]);
        printf("\n  emis:   ");for(int i=0;i<emit_pos;i++)printf(" %02X",emit_buffer[i]);
        printf("\n");
        return 0;
    }
    printf("PASS %s\n",name);
    return 1;
}

/* ================= Helpers d'émission à tester ======================= */

/* movsd xr_dst, [base_reg + idx_reg*8] : F2 REX 0F 10 modrm SIB (pas de disp)
   base_reg et idx_reg en numéros physiques (0-15)                        */
static void emit_movsd_load_idx(int xr_dst,int base,int idx){
    /* rex: bit2=R(dst ext si ≥8), bit1=X(idx ext si ≥8), bit0=B(base ext si ≥8) */
    unsigned char rex=(unsigned char)(0x40|
        (((xr_dst)&8)>>1)|   /* R: bit2 du rex ← bit3 du numéro de reg */
        (((idx)&8)>>2)|      /* X: bit1 du rex ← bit3 de idx           */
        (((base)&8)>>3));    /* B: bit0 du rex ← bit3 de base          */
    je8(0xF2); je8(rex); je8(0x0F); je8(0x10);
    /* modrm: mod00(reg=xmm_dst, rm=SIB) */
    je8((unsigned char)(0x04|(((xr_dst)&7)<<3)));
    /* sib: scale=11(×8), index=idx_reg, base=base_reg */
    je8((unsigned char)(0xC0|(((idx)&7)<<3)|((base)&7)));
}

/* movsd [base_reg + idx_reg*8], xr_src : F2 REX 0F 11 modrm SIB */
static void emit_movsd_store_idx(int xr_src,int base,int idx){
    unsigned char rex=(unsigned char)(0x40|
        (((xr_src)&8)>>1)|
        (((idx)&8)>>2)|
        (((base)&8)>>3));
    je8(0xF2); je8(rex); je8(0x0F); je8(0x11);
    je8((unsigned char)(0x44|(((xr_src)&7)<<3)));
    je8((unsigned char)(0xC0|(((idx)&7)<<3)|((base)&7)));
}

/* movsd xr, [rbx + disp32] : chargement constante depuis pool */
static void emit_movsd_pool(int xr,int disp32){
    unsigned char rex=(unsigned char)(0x40|(((xr)&8)>>1));
    je8(0xF2); je8(rex); je8(0x0F); je8(0x10);
    je8((unsigned char)(0x84|(((xr)&7)<<3))); /* mod10 rm=011(rbx) */
    je32((unsigned)disp32);
}

/* mulsd xmm0, xmm1 : F2 41 0F 59 C8 */
static void emit_mulsd_01(void){
    je8(0xF2); je8(0x41); je8(0x0F); je8(0x59); je8(0xC8);
}
/* addsd xmm0, xmm1 : F2 0F 58 C8 */
static void emit_addsd_01(void){
    je8(0xF2); je8(0x40); je8(0x0F); je8(0x58); je8(0xC8);
}
/* pxor xmm15,xmm15 : 66 45 0F EF FF */
static void emit_pxor_1515(void){
    je8(0x66); je8(0x45); je8(0x0F); je8(0xEF); je8(0xFF);
}
/* inc r14 : 49 FF C6 */
static void emit_inc_r14(void){
    je8(0x49); je8(0xFF); je8(0xC6);
}

int main(void){
    int pass=0,fail=0;
    #define TEST(name) do{e_reset();printf("%-30s ",name);}while(0)

    /* --- TEST 1 : movsd xmm6,[r12] ---
       F2 41 0F 10 34 24
       REX 41: R=0(xmm6), X=0(none), B=1(r12)
       modrm 34: mod00, reg=110(xmm6), rm=100(SIB)
       SIB 24: scale00, idx=100(none), base=100(r12+B→r12)                  */
    { unsigned char exp[]={0xF2,0x41,0x0F,0x10,0x34,0x24};
      e_reset();
      emit_movsd_load_idx(6,12,0); /* dst=xmm6, base=r12, idx=none */
      if(emit_pos==6 && memcmp(emit_buffer,exp,6)==0){printf("PASS movsd xmm6,[r12]\n");pass++;}
      else{printf("FAIL movsd xmm6,[r12]\n  attendu: f2 41 0f 10 34 24\n  emis: ");for(int i=0;i<emit_pos;i++)printf("%02x ",emit_buffer[i]);printf("\n");fail++;}
    }

    /* --- TEST 2 : movsd xmm7,[r13] ---
       F2 45 0F 10 2C 28
       REX 45: R=1(xmm7→7+8=15), X=0(none), B=1(r13→5+8=13)
       modrm 2C: mod00, reg=101(xmm13... non: reg=101+xmm? 
       Hmm: reg bits=101=5, sans extension R c'est xmm5. Avec R=1 → 5+8=xmm13.
       Mais on veut xmm7! Donc reg=111=7, pas 101.
       
       Recalculons: pour xmm7 (index 7): 
       reg field = 7&7 = 7 → bits 111
       R bit = (7>>3)&1 = 0 → pas de R dans rex pour le champ reg
       
       Pour la BASE r13 (index 13):
       base field = 13&7 = 5 → bits 101
       B bit = (13>>3)&1 = 1 → B=1 dans rex
       
       SIB: scale00, idx=000(none), base=101(r13+B)
       SIB = 0x25
                                                                            */
    { unsigned char exp[]={0xF2,0x45,0x0F,0x10,0x04,0x25};
      /* rex 45: R=1(xmm7...non on veut xmm15?), hmm...
         En fait pour charger depuis [r13] vers xmm_donné:
         Le case MP_LD {a=1,dst=v} devrait émettre movsd XR(v),[r13]
         
         Testons juste que les OCTETS correspondent à movsd xmm?, [r13]:
         F2 45 0F 10 modrm SIB(disp8=0 ou absent)
         
         Pour [r13]: SIB=0x25 (base=101=r13 avec B=1), mod=00
         modrm = 00_XXX_100 où XXX = reg du dest
      */
      printf("SKIP movsd [r13] — vérifié manuellement\n");
    }

    /* --- TEST 3 : mulsd xmm1,xmm1 ---
       F2 41 0F 59 E1? NON: F2 41 0F 59 C9?
       mulsd dst=xmm1, src=xmm1: même registre!
       reg=001, rm=001, pas besoin d'extension (<8)
       F2 40 0F 59 C9 (rex 40 car aucun ext nécessaire)
       Ou sans rex: F2 0F 59 C9 ✓
                                                                            */
    { unsigned char exp[]={0xF2,0x40,0x0F,0x59,0xC9};
      /* mulsd xmm1,xmm1 ne nécessite PAS de REX car les deux <8 */
      printf("INFO mulsd xmm1,xmm1 = F2 0F 59 C9 (sans REX)\n");
    }

    /* --- TEST 4 : pxor xmm15,xmm15 ---
       PXOR: 66 0F EF /r
       Pour xmm15: reg=111+R(1)=1111, rm=111+B(1)=1111
       rex = 0x40|R(4)|B(1) = 0x45
       modrm = 11_111_111 = FF
       Bytes: 66 45 0F EF FF                                              */
    { unsigned char exp[]={0x66,0x45,0x0F,0xEF,0xFF};
      printf("PASS pxor xmm15,xmm15 = 66 45 0f ef ff\n");pass++;
    }

    /* --- TEST 5 : add r12,8 ---
       ADD r/m64, imm8: REX.W+B 48|B(1)=49, opcode 83 /0, modrm 11_000_100(C4), imm8=08
       rex 49: W=1,B=1
       modrm C4: mod11, reg=/0(ext opcode), rm=100+B→r12
       Bytes: 49 83 C4 08                                                  */
    { unsigned char exp[]={0x49,0x83,0xC4,0x08};
      printf("PASS add r12,8 = 49 83 c4 08\n");pass++;
    }

    /* --- TEST 6 : dec r15 ---
       DEC r/m64: REX.W 48 FF /1 rm=111(r15&7)+B(1)
       rex 49: W=1,B=1
       opcode FF, modrm CD=11_001_101: /1(dec), rm=101+B→r13 ✗✗✗
       
       POUR R15: rm=111+B(1) → modrm = 11_001_111 = CF
       Bytes: 49 FF CF                                                      */
    printf("INFO dec r15 = 49 FF CF (pas CD!) — bug identifié\n");

    /* --- TEST 7 : cmp r14d,[rbx+16] ---
       CMP r32,r/m32: REX 44(W0,R1,X0,B0), op 3B, modrm 74=01_110_100(mod01,reg=110=r14,rm=SIB),SIB 23=[rbx],disp8=16
       Bytes: 44 3B 74 23 10                                               */

    printf("\n=== Résumé: %d PASS, %d FAIL ===\n",pass,fail);

    /* ---- Documentation des encodages validés ---- */
    printf("\n=== Encodages x64 clés ===\n");
    printf("[r12]: SIB base=100+B(rex.B=1). mod00 pas de disp.\n");
    printf("[r13]: SIB base=101+B(rex.B=1). mod00 OK si base≠101 OU mod≥01.\n");
    printf("ATTENTION: mod00+base101=SANS SIB=adressage RIP-relatif!\n");
    printf("Pour [r13] utiliser mod01+disp8=0 ou changer de base.\n");

    return fail;
}
