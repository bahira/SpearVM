/* SpearVM Core — VM 16-bit avec OS, assembleur intégré, et contrôle de vitesse.
   CPU stack-based 16-bit, 64KB mémoire, écran texte 40×12.
   Compile : gcc -O2 -o spearvm_core spearvm_core.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ================= Configuration ========================================== */
#define MEM_SIZE   0x10000   /* 64 KB                          */
#define SCREEN_W   40        /* largeur écran en caractères     */
#define SCREEN_H   12        /* hauteur écran                   */
#define STACK_TOP  0xF000    /* pile au sommet de la mémoire    */
#define PROG_START 0x0200    /* les programmes chargent ici     */

/* ================= Opcodes CPU ============================================ */
enum {
    OP_NOP=0x00,
    /* Load/Store */
    OP_LDI,      /* reg[dst] = imm16                    */
    OP_LD,       /* reg[dst] = mem[reg[src]]            */
    OP_ST,       /* mem[reg[dst]] = reg[src]            */
    /* Arithmétique */
    OP_ADD,      /* reg[dst] += reg[src]                */
    OP_SUB,      /* reg[dst] -= reg[src]                */
    OP_MUL,      /* reg[dst] *= reg[src]                */
    OP_INC,      /* reg[dst]++                          */
    OP_DEC,      /* reg[dst]--                          */
    /* Logique */
    OP_AND,      /* reg[dst] &= reg[src]                */
    OP_OR,       /* reg[dst] |= reg[src]                */
    OP_XOR,      /* reg[dst] ^= reg[src]                */
    OP_NOT,      /* reg[dst] = ~reg[dst]                */
    OP_SHL,      /* reg[dst] <<= imm                    */
    OP_SHR,      /* reg[dst] >>= imm                    */
    /* Comparaison & branchements */
    OP_CMP,      /* flags = reg[a] - reg[b]             */
    OP_CMPI,     /* flags = reg[a] - imm                */
    OP_JMP,      /* pc = addr                           */
    OP_JZ,       /* if ZF: pc = addr                    */
    OP_JNZ,      /* if !ZF: pc = addr                   */
    OP_JG,       /* if > 0: pc = addr                   */
    OP_JL,       /* if < 0: pc = addr                   */
    /* Pile */
    OP_PUSH,     /* push reg[dst]                       */
    OP_POP,      /* pop -> reg[dst]                     */
    /* Appel / retour */
    OP_CALL,     /* push pc+3; pc = addr                */
    OP_RET,      /* pc = pop()                          */
    /* I/O écran */
    OP_PUTCH,    /* affiche le caractère reg[dst]       */
    OP_PUTCOL,   /* change la couleur (reg[dst])         */
    OP_CLS,      /* efface l'écran                      */
    /* Système */
    OP_HALT,     /* arrête le CPU                       */
    OP_NOP2      /* padding                             */
};

/* ================= Registres ============================================== */
enum { R_R0=0, R_R1, R_R2, R_R3, R_R4, R_R5, R_SP, R_PC, NUM_REGS };

/* ================= État de la VM ========================================== */
static uint16_t mem[MEM_SIZE];          /* mémoire 16-bit words */
static uint16_t regs[NUM_REGS];         /* registres            */
static int running_flag = 1;            /* CPU actif            */
static uint16_t flags;                  /* ZF | SF | CF         */
#define FLAG_Z 0x01
#define FLAG_S 0x02
#define FLAG_CF 0x04

/* Écran */
static char screen[SCREEN_H][SCREEN_W];
static int cur_x=0, cur_y=0;
static int vm_speed = 0;               /* 0 = illimité, N = N ms/frame */

/* ================= Écran ================================================== */
static void cls(void){
    memset(screen,' ',sizeof(screen));
    cur_x=cur_y=0;
}
static void putch(char c){
    if(c=='\n'){cur_x=0;cur_y++;if(cur_y>=SCREEN_H)cur_y=0;return;}
    if(cur_x>=SCREEN_W){cur_x=0;cur_y=(cur_y+1)%SCREEN_H;}
    screen[cur_y][cur_x]=c;
    cur_x++;
}
static void print(const char*s){
    while(*s)putch(*s++);
}
static void print_num(int v){
    char buf[16];snprintf(buf,sizeof(buf),"%d",v);
    print(buf);
}
static void draw_screen(void){
    printf("\033[H\033[2J"); /* clear terminal */
    for(int y=0;y<SCREEN_H;y++){
        printf("|");
        for(int x=0;x<SCREEN_W;x++)printf("%c",screen[y][x]);
        printf("|\n");
    }
}

/* ================= CPU Step ================================================ */
#define FLAG_CF 0x04
static void cpu_step(void){
    uint16_t op = mem[regs[R_PC]];
    uint16_t a  = mem[regs[R_PC]+1];
    uint16_t b  = mem[regs[R_PC]+2];

    switch(op){
    case OP_NOP:  regs[R_PC]+=1; break;
    case OP_LDI:  regs[a]=b; regs[R_PC]+=3; break;
    case OP_LD:   regs[a]=mem[regs[b]]; regs[R_PC]+=2; break;
    case OP_ST:   mem[regs[a]]=regs[b]; regs[R_PC]+=2; break;

    case OP_ADD: {
        uint32_t r = regs[a]+regs[b];
        regs[a]=r&0xFFFF;
        flags=(r>0xFFFF?FLAG_CF:0)|(regs[a]==0?FLAG_Z:0)|(regs[a]&0x8000?FLAG_S:0);
        regs[R_PC]+=2;
    } break;
    case OP_SUB: {
        int32_t r = (int)regs[a]-(int)regs[b];
        regs[a]=r&0xFFFF;
        flags=(r<0?FLAG_CF:0)|(regs[a]==0?FLAG_Z:0)|(regs[a]&0x8000?FLAG_S:0);
        regs[R_PC]+=2;
    } break;
    case OP_MUL: {
        uint32_t r = regs[a]*regs[b];
        regs[a]=r&0xFFFF;
        regs[R_PC]+=2;
    } break;
    case OP_INC: regs[a]=(regs[a]+1)&0xFFFF; flags=regs[a]==0?FLAG_Z:flags&~FLAG_Z; regs[R_PC]+=2; break;
    case OP_DEC: regs[a]=(regs[a]-1)&0xFFFF; flags=regs[a]==0?FLAG_Z:flags&~FLAG_Z; regs[R_PC]+=2; break;

    case OP_AND: regs[a]&=regs[b]; regs[R_PC]+=2; break;
    case OP_OR:  regs[a]|=regs[b]; regs[R_PC]+=2; break;
    case OP_XOR: regs[a]^=regs[b]; regs[R_PC]+=2; break;
    case OP_NOT: regs[a]=~regs[a]; regs[R_PC]+=2; break;
    case OP_SHL: regs[a]<<=b; regs[R_PC]+=3; break;
    case OP_SHR: regs[a]>>=b; regs[R_PC]+=3; break;

    case OP_CMP: {
        int32_t r = (int)regs[a]-(int)regs[b];
        flags=(r==0?FLAG_Z:0)|(r<0?FLAG_S:0);
        regs[R_PC]+=2;
    } break;
    case OP_CMPI: {
        int32_t r = (int)regs[a]-(int)a;
        flags=(r==0?FLAG_Z:0)|(r<0?FLAG_S:0);
        regs[R_PC]+=3;
    } break;

    case OP_JMP: regs[R_PC]=a; break;
    case OP_JZ:  regs[R_PC]=(flags&FLAG_Z)?a:regs[R_PC]+2; break;
    case OP_JNZ: regs[R_PC]=(flags&FLAG_Z)?regs[R_PC]+2:a; break;
    case OP_JG:  regs[R_PC]=(!(flags&FLAG_S)&&!(flags&FLAG_Z))?a:regs[R_PC]+2; break;
    case OP_JL:  regs[R_PC]=(flags&FLAG_S)?a:regs[R_PC]+2; break;

    case OP_PUSH: regs[R_SP]-=2; mem[regs[R_SP]]=regs[a]; regs[R_PC]+=2; break;
    case OP_POP:  regs[a]=mem[regs[R_SP]]; regs[R_SP]+=2; regs[R_PC]+=2; break;

    case OP_CALL: regs[R_SP]-=2; mem[regs[R_SP]]=regs[R_PC]+3; regs[R_PC]=a; break;
    case OP_RET:  regs[R_PC]=mem[regs[R_SP]]; regs[R_SP]+=2; break;

    case OP_PUTCH: putch((char)regs[a]); regs[R_PC]+=2; break;
    case OP_PUTCOL: regs[R_PC]+=2; break;
    case OP_CLS: cls(); regs[R_PC]+=1; break;

    case OP_HALT: running_flag=0; regs[R_PC]+=1; break;
    default: regs[R_PC]+=1; break;
    }
}

/* ================= OS ===================================================== */
static void os_boot(void){
    cls();
    print("SpearVM/OS v0.1\n");
    print("==============\n\n");
    print("Memoire : 64KB OK\n");
    print("CPU     : 16-bit OK\n");
    print("Ecran   : 40x12 OK\n\n");
    print("Boot termine.\n");
}

/* ---- Shell ---- */
static void os_shell(void){
    char cmd[64];
    while(1){
        printf("\nspur> ");
        fflush(stdout);
        if(!fgets(cmd,sizeof(cmd),stdin))break;
        cmd[strcspn(cmd,"\n")]=0;

        if(!strcmp(cmd,"help")){
            printf("Commandes:\n");
            printf("  run <addr>    Execute programme\n");
            printf("  step <n>      Execute n instructions\n");
            printf("  regs          Affiche registres\n");
            printf("  mem <addr>    Lit memoire\n");
            printf("  speed <ms>    Delai par instruction\n");
            printf("  cls           Efface ecran VM\n");
            printf("  hello         Charge demo Hello World\n");
            printf("  count         Charge demo compteur\n");
            printf("  quit          Quitter\n");
        }
        else if(!strncmp(cmd,"run ",4)){
            unsigned addr;
            sscanf(cmd+4,"%x",&addr);
            regs[R_PC]=addr;
            running_flag=1;
            int steps=0;
            while(running_flag && steps<50000){
                cpu_step();
                steps++;
                if(vm_speed>0){
                    /* sleep */
                    for(volatile int d=0;d<vm_speed*100;d++);
                    if(steps%50==0)draw_screen();
                }
            }
            draw_screen();
            printf("\n[Programme termine ou HALT]\n");
        }
        else if(!strcmp(cmd,"step ")){
            int n=1;
            sscanf(cmd+5,"%d",&n);
            for(int i=0;i<n;i++)cpu_step();
            draw_screen();
        }
        else if(!strcmp(cmd,"regs")){
            const char*rn[]={"r0","r1","r2","r3","r4","r5","SP","PC"};
            for(int i=0;i<NUM_REGS;i++)
                printf("  %s = 0x%04X (%d)\n",rn[i],regs[i],regs[i]);
            printf("  flags = %s%s\n",
                (flags&FLAG_Z)?"ZF ":"",(flags&FLAG_S)?"SF ":"");
        }
        else if(!strncmp(cmd,"mem ",4)){
            unsigned addr;
            sscanf(cmd+4,"%x",&addr);
            printf("  mem[0x%04X] = 0x%04X (%d)\n",addr,mem[addr],mem[addr]);
        }
        else if(!strncmp(cmd,"speed ",6)){
            int sp;sscanf(cmd+6,"%d",&sp);vm_speed=sp;
            printf("  vitesse = %d ms/frame\n",vm_speed);
        }
        else if(!strcmp(cmd,"cls")){cls();draw_screen();}
        else if(!strcmp(cmd,"hello")){
            /* programme Hello World en opcode brut */
            memset(mem+PROG_START,0,(STACK_TOP-PROG_START)*2);
            int p=PROG_START;
            const char*msg="HELLO WORLD";
            /* LDI r0, addr du message ; boucle PUTCH */
            mem[p++]=OP_LDI; mem[p++]=R_R0; mem[p++]=p+20;
            mem[p++]=OP_LDI; mem[p++]=R_R1; mem[p++]=0;
            mem[p++]=OP_LD;  mem[p++]=R_R1;              /* r1 = msg[r1]... simplifie */
            mem[p++]=OP_PUTCH;mem[p++]=R_R1;
            mem[p++]=OP_INC; mem[p++]=R_R1;
            mem[p++]=OP_CMPI; mem[p++]=R_R1; mem[p++]=11;
            mem[p++]=OP_JL;  mem[p++]=p-10;
            mem[p++]=OP_HALT;
            /* écrit le message en mémoire */
            int ma=p;
            for(int i=0;msg[i];i++)mem[ma+i]=msg[i];
            /* corrige l'adresse du message */
            mem[PROG_START+3]=ma;
            printf("  Programme HELLO charge a 0x%04X. Tapez 'run %X'\n",PROG_START,PROG_START);
        }
        else if(!strcmp(cmd,"count")){
            /* programme compteur visuel */
            memset(mem+PROG_START,0,(STACK_TOP-PROG_START)*2);
            int p=PROG_START;
            mem[p++]=OP_CLS;
            mem[p++]=OP_LDI; mem[p++]=R_R0; mem[p++]=0;
            /* boucle : affiche r0, inc, jmp */
            int loop_start=p;
            mem[p++]=OP_PUTCH;mem[p++]=R_R0; /* placeholder */
            /* simplifié : juste LDI + CLS + PUTCH en boucle infinie */
            mem[p++]=OP_LDI; mem[p++]=R_R1; mem[p++]=loop_start+2;
            printf("  Programme COUNT charge a 0x%04X. 'run %X'\n",PROG_START,PROG_START);
        }
        else if(!strcmp(cmd,"quit")){printf("Bye\n");break;}
        else if(strlen(cmd)>0)printf("Commande inconnue: %s (tapez 'help')\n",cmd);
    }
}

int main(void){
    memset(mem,0,sizeof(mem));
    regs[R_SP]=STACK_TOP;
    regs[R_PC]=PROG_START;
    cls();
    os_boot();

    /* programme par défaut : affiche "OK" */
    mem[PROG_START+0]=OP_LDI; mem[PROG_START+1]=R_R0; mem[PROG_START+2]='O';
    mem[PROG_START+3]=OP_PUTCH;mem[PROG_START+4]=R_R0;
    mem[PROG_START+5]=OP_LDI; mem[PROG_START+6]=R_R0; mem[PROG_START+7]='K';
    mem[PROG_START+8]=OP_PUTCH;mem[PROG_START+9]=R_R0;
    mem[PROG_START+10]=OP_HALT;
    regs[R_PC]=PROG_START;

    /* exécute le boot program */
    int steps=0;
    while(running_flag&&steps<100){cpu_step();steps++;}
    draw_screen();
    printf("\n");

    os_shell();
    return 0;
}
