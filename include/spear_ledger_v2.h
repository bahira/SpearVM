/* spear_ledger_v2.h — noyaux SPEAR v2 genere par fitting rationnel Lawson.
   Genere automatiquement par scripts/fit_spear_v2.py — NE PAS EDITER.
   Precisions mesurees (double, Python ref IEEE) :
     tanh_v2    max_err 6.704e-04 sur [-6,6]
     erf_v2     max_err 2.337e-05 sur [-6,6]
     gelu_v2    max_err 6.269e-03 sur [-6,6]
     sigmoid_v2 max_err 1.204e-04 sur [-6,6]
   Formes : impaires rationnelles x*N(y)/D(y), y=x^2 ; queues saturees. */
#ifndef SPEAR_LEDGER_V2_H
#define SPEAR_LEDGER_V2_H

/* ---- tanh_v2 : T(x) = x*(n0 + n1*y + ... )/(1 + d1*y + ...), y = x*x ---- */
#define SPV2_TANH_NUM { 9.99671553747818797e-01, 9.69818098308345700e-02, 5.34898030509189373e-04 }
#define SPV2_TANH_DEN { 1.00000000000000000e+00, 4.28929626818397136e-01, 1.13140132579288184e-02 }
#define SPV2_TANH_NNUM 3
#define SPV2_TANH_NDEN 3
/* saturation de queue |x| > 4 */

/* ---- erf_v2 : meme forme paire---------------------------------------- */
#define SPV2_ERF_NUM { 1.12841751266903279e+00, 1.83482771948230095e-01, 5.73373674730976793e-02, 2.48430060206610405e-03, 3.72785350475749968e-06 }
#define SPV2_ERF_DEN { 1.00000000000000000e+00, 4.96471589671860558e-01, 1.14910282096263028e-01, 1.61717422205343367e-02, 1.86656477609649336e-04, -1.74401807407079551e-07 }
#define SPV2_ERF_NNUM 5
#define SPV2_ERF_NDEN 6
/* saturation de queue |x| > 3.5 */

/* ---- gelu_v2 constantes ----------------------------------------------- */
#define SPV2_GELU_C1 7.97884560802865406e-01
#define SPV2_GELU_C3 0.044715

#endif /* SPEAR_LEDGER_V2_H */
