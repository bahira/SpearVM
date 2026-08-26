# -*- coding: utf-8 -*-
# Applique les 3 fixes au src/spur.c restaure depuis git
p = "src/spur.c"
d = open(p, encoding="utf-8").read()

# FIX 1: SUBSD -> MINSD dans le clamp haut de rat_on_xmm0
old1 = "e_movsd_pool(1,hiK); e_sd(0x5A,0,1);      /* min(y,hi)"
new1 = "e_movsd_pool(1,hiK); e_sd(0x5D,0,1);      /* min(y,hi)"
assert old1 in d, "fix1: pattern absent"
d = d.replace(old1, new1)

# FIX 2: ajoute pxor xmm14 (zero accumulateur) apres pxor xmm15
old2 = "pxor xmm15,xmm15"
new2 = "pxor xmm15,xmm15\n    je8(0x66);je8(0x43);je8(0x0F);je8(0xEF);je8(0xF6); /* pxor xmm14 */"
assert old2 in d, "fix2: pattern absent"
d = d.replace(old2, new2, 1)

# FIX 3: MP_ST rex 0x42 -> 0x41 (retire le bit X fantome)
old3 = "je8((unsigned char)(0x42|(((xr)&8)>>1)));"
new3 = "je8((unsigned char)(0x41|(((xr)&8)>>1)));"
assert old3 in d, "fix3: pattern absent"
d = d.replace(old3, new3)

open(p, "w", encoding="utf-8").write(d)
print("3 fixes appliques OK")
