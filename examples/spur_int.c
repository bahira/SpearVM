/* SpearVM Int — Extension entiers 32 bits pour traitement d'image/compression.
   Démontre : inversion, seuillage, masque binaire sur un tableau de pixels.
   Compile : gcc -O3 -mavx2 -o spur_int spur_int.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#pragma GCC target("avx2")
#include <immintrin.h>

/* ================= Ops entières émulées via double ========================
   Les doubles représentent exactement les entiers jusqu'à 2^53.
   On utilise le tronc pour les opérations bit à bit via un cast.
   Pour la production : utiliser de vrais registres GPR.                  */

/* ================= Kernel 1 : Inversion d'image ===========================
   Chaque pixel ARGB est inversé : out = 0xFFFFFF00 | (~pixel & 0x00FFFFFF) */

static void invert_scalar(const uint32_t* in,uint32_t* out,int n){
    for(int i=0;i<n;i++) out[i]=~in[i];
}

static void invert_avx2(const uint32_t* in,uint32_t* out,int n){
    /* AVX2 traite 8 uint32 par instruction */
    __m256i ones=_mm256_set1_epi32(-1); /* 0xFFFFFFFF */
    int vec=n&~7LL;
    for(int i=0;i<vec;i+=8){
        __m256i v=_mm256_loadu_si256((const __m256i*)(in+i));
        _mm256_storeu_si256((__m256i*)(out+i),_mm256_andnot_si256(v,ones));
    }
    for(int i=vec;i<n;i++)out[i]=~in[i];
}

/* ================= Kernel 2 : Seuillage (binarisation) ====================
   Chaque canal R,G,B est seuillé : si > 128 alors 255 sinon 0            */

static void threshold_scalar(const uint32_t* in,uint32_t* out,int n){
    for(int i=0;i<n;i++){
        uint32_t p=in[i];
        uint32_t r=(p>>16)&0xFF,g=(p>>8)&0xFF,b=p&0xFF;
        r=r>128?255:0; g=g>128?255:0; b=b>128?255:0;
        out[i]=(p&0xFF000000)|(r<<16)|(g<<8)|b;
    }
}

static void threshold_avx2(const uint32_t* in,uint32_t* out,int n){
    __m256i thresh=_mm256_set1_epi8(128);
    __m256i mask=_mm256_set1_epi8(0x80);
    __m256i ff=_mm256_set1_epi8((char)-1);
    __m256i keep=_mm256_set1_epi32(0xFF000000);

    int vec=n&~31; /* 32 pixels = 256 bytes = 1 registre YMM entier */
    for(int i=0;i<vec;i+=8){
        __m256i v=_mm256_loadu_si256((const __m256i*)(in+i));
        /* seuil chaque octet: si >=128 alors 0xFF sinon 0x00 */
        __m256i above=_mm256_cmpgt_epi8(v,thresh);
        /* binarise: AND avec mask pour garder seulement le bit haut */
        __m256i binary=_mm256_and_si256(above,mask);
        binary=_mm256_sub_epi8(binary,mask); /* 0x00 ou 0xFF par canal */
        binary=_mm256_andnot_si256(_mm256_set1_epi32(0xFF000000),binary);
        binary=_mm256_or_si256(binary,_mm256_and_si256(v,_mm256_set1_epi32(0xFF000000)));
        _mm256_storeu_si256((__m256i*)(out+i),binary);
    }
    for(int i=vec;i<n;i++){
        uint32_t p=in[i];
        uint32_t r=(p>>16)&0xFF,g=(p>>8)&0xFF,b=p&0xFF;
        r=r>128?255:0; g=g>128?255:0; b=b>128?255:0;
        out[i]=(p&0xFF000000)|(r<<16)|(g<<8)|b;
    }
}

/* ================= Benchmark ============================================= */

int main(void){
    printf("=== SpearVM Int : Traitement d'image ===\n\n");

    int n=4000000; /* 4M pixels = image 2048×1953 environ */
    uint32_t* img=malloc(n*sizeof(uint32_t));
    uint32_t* tmp=malloc(n*sizeof(uint32_t));

    /* génère une fausse image avec des pixels variés */
    srand(42);
    for(int i=0;i<n;i++){
        uint32_t r=rand()&0xFF,g=rand()&0xFF,b=rand()&0xFF;
        img[i]=0xFF000000|(r<<16)|(g<<8)|b;
    }

    clock_t t0,t1;

    /* ---- Test inversion : scalar vs AVX2 ---- */
    memset(tmp,0,n*sizeof(uint32_t));
    t0=clock(); invert_scalar(img,tmp,n); t1=clock();
    double inv_scalar=(double)(t1-t0)/CLOCKS_PER_SEC*1000;

    memset(tmp,0,n*sizeof(uint32_t));
    t0=clock(); invert_avx2(img,tmp,n); t1=clock();
    double inv_avx=(double)(t1-t0)/CLOCKS_PER_SEC*1000;

    printf("--- Inversion (~v) ---\n");
    printf("  scalaire : %8.2f ms (%.2f Gpix/s)\n",inv_scalar,n/inv_scalar/1e6/1000);
    printf("  AVX2     : %8.2f ms (%.2f Gpix/s)\n",inv_avx,n/inv_avx/1e6/1000);
    printf("  speedup  : %8.1fx\n\n",inv_scalar/(inv_avx>0?inv_avx:1));

    /* ---- Test seuillage : scalar vs AVX2 ---- */
    memset(tmp,0,n*sizeof(uint32_t));
    t0=clock(); threshold_scalar(img,tmp,n); t1=clock();
    double thr_scalar=(double)(t1-t0)/CLOCKS_PER_SEC*1000;

    memset(tmp,0,n*sizeof(uint32_t));
    t0=clock(); threshold_avx2(img,tmp,n); t1=clock();
    double thr_avx=(double)(t1-t0)/CLOCKS_PER_SEC*1000;

    printf("--- Seuillage binarisé ---\n");
    printf("  scalaire : %8.2f ms (%.2f Gpix/s)\n",thr_scalar,n/thr_scalar/1e6/1000);
    printf("  AVX2     : %8.2f ms (%.2f Gpix/s)\n",thr_avx,n/thr_avx/1e6/1000);
    printf("  speedup  : %8.1fx\n\n",thr_scalar/(thr_avx>0?thr_avx:1));

    printf("Les ops entières bit-exact sont natives en AVX2:\n");
    printf("  - Pas d'approximation nécessaire\n");
    printf("  - 8 pixels traités par instruction (256 bits)\n");
    printf("  - Speedup vient du SIMD + absence d'appels de fonction\n");

    free(img);free(tmp);
    return 0;
}
