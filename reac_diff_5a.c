/*
 Multi-component Lattice Boltzmann simulation in which passive scalar chemical species and thermal energy
 are all advected by the fluid, which undergoes buoyancy-driven natural convection
 Created by Stuart Bartlett on 31/07/13.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <float.h>

#define i_3  0.33333333333333
#define t_3  0.66666666666667
#define i_6  0.16666666666667
#define i_9  0.11111111111111
#define f_9  0.44444444444444
#define i_12 0.083333333333333
#define i_36 0.027777777777778
#define t_i7 1.7142857142857
#define rm_36 0.027777777777778
#define pi 3.14159265359

// Constants for random number generator (sourse: Numerical Recipes)
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

long seed;

// Random number generator

float ran2(long *idum)
{
    int j;
    long k;
    static long idum2=123456789;
    static long iy=0;
    static long iv[NTAB];
    float temp;
    
    if (*idum <= 0) {
        if (-(*idum) < 1) *idum=1;
        else *idum = -(*idum);
        idum2=(*idum);
        for (j=NTAB+7;j>=0;j--) {
            k=(*idum)/IQ1;
            *idum=IA1*(*idum-k*IQ1)-k*IR1;
            if (*idum < 0) *idum += IM1;
            if (j < NTAB) iv[j] = *idum;
        }
        iy=iv[0];
    }
    k=(*idum)/IQ1;
    *idum=IA1*(*idum-k*IQ1)-k*IR1;
    if (*idum < 0) *idum += IM1;
    k=idum2/IQ2;
    idum2=IA2*(idum2-k*IQ2)-k*IR2;
    if (idum2 < 0) idum2 += IM2;
    j=iy/NDIV;
    iy=iv[j]-idum2;
    iv[j] = *idum;
    if (iy < 1) iy += IMM1;
    if ((temp=AM*iy) > RNMX) return RNMX;
    else return temp;
}

// All distribution functions move along their respective velocity vectors (streaming step)
void bb_per_bc(double *f,double *f_0,int t_cs,int div,int y_s,int l_1,int l_2,int n_cs){
    int i,j,k;
    for(k=0;k<(2+n_cs)*9;k+=9){
        // Horizontally pointing distributions simply move as a block
        memcpy(&f[(k+1)*t_cs+y_s],&f_0[(k+1)*t_cs],l_2);            // Move f_1s to the right
        memcpy(&f[(k+3)*t_cs],&f_0[(k+3)*t_cs+y_s],l_2);            // Move f_3s to the left
        // Now correct the positions of the exchanged distributions (those with diagonal unit vectors)
        memmove(&f[(k+5)*t_cs],&f[(k+5)*t_cs+1],l_1);				// Move leftmost column of f_5s up
        memmove(&f[(k+7)*t_cs-y_s],&f[(k+7)*t_cs-y_s+1],l_1);		// Move rightmost column of f_6s up
        memmove(&f[(k+8)*t_cs-y_s+1],&f[(k+8)*t_cs-y_s],l_1);		// Move rightmost column of f_7s down
        memmove(&f[(k+8)*t_cs+1],&f[(k+8)*t_cs],l_1);				// Move leftmost column of f_8s down
        // Other distributions must be moved one column at a time
        for(j=0;j<div;j++){
            memcpy(&f[(k+2)*t_cs+j*y_s],&f_0[(k+2)*t_cs+j*y_s+1],l_1);			// Move this column of f_2s up
            memcpy(&f[(k+4)*t_cs+j*y_s+1],&f_0[(k+4)*t_cs+j*y_s],l_1);			// Move this column of f_4s down
            if(j<div-1){
                memcpy(&f[(k+5)*t_cs+(j+1)*y_s],&f_0[(k+5)*t_cs+j*y_s+1],l_1);	// Move this column of f_5s up and to the right
                memcpy(&f[(k+6)*t_cs+j*y_s],&f_0[(k+6)*t_cs+(j+1)*y_s+1],l_1);	// Move this column of f_6s up and to the left
                memcpy(&f[(k+7)*t_cs+j*y_s+1],&f_0[(k+7)*t_cs+(j+1)*y_s],l_1);	// Move this column of f_7s down and to the left
                memcpy(&f[(k+8)*t_cs+(j+1)*y_s+1],&f_0[(k+8)*t_cs+j*y_s],l_1);	// Move this column of f_8s down and to the right
            }
            f[(k+2)*t_cs+(j+1)*y_s-1]=f_0[(k+4)*t_cs+(j+1)*y_s-1];      // f_4s bounce back to become f_2s
            f[(k+4)*t_cs+j*y_s]=f_0[(k+2)*t_cs+j*y_s];					// f_2s bounce back to become f_4s
            f[(k+5)*t_cs+(j+1)*y_s-1]=f_0[(k+7)*t_cs+(j+1)*y_s-1];      // f_7s bounce back to become f_5s
            f[(k+6)*t_cs+(j+1)*y_s-1]=f_0[(k+8)*t_cs+(j+1)*y_s-1];      // f_8s bounce back to become f_6s
            f[(k+7)*t_cs+j*y_s]=f_0[(k+5)*t_cs+j*y_s];					// f_5s bounce back to become f_7s
            f[(k+8)*t_cs+j*y_s]=f_0[(k+6)*t_cs+j*y_s];					// f_6s bounce back to become f_8s
        }
    }
}

// Calculate densities and temperatures
void calc_dens(double *f,double *rho,double *eps,double *psi,double T_a,double T_b,double *phi,int y_s,
               int div,int n_cs,int t_cs,int fc_s,int grad,double T_s,double F,double x_fac,int n_ss,int rank,
               int num_procs,int x_s,double *omega){
    int i,j,k,l;
    double en_def,T_0,i_rho,x,y,dis;
    
    // Calculate all densities and internal energies
    for(i=0;i<t_cs;i++){
        rho[i]=f[i]+f[t_cs+i]+f[2*t_cs+i]+f[3*t_cs+i]+f[4*t_cs+i]+f[5*t_cs+i]+f[6*t_cs+i]+f[7*t_cs+i]+f[8*t_cs+i];
        i_rho=1/rho[i];
        eps[i]=(f[9*t_cs+i]+f[10*t_cs+i]+f[11*t_cs+i]+f[12*t_cs+i]+f[13*t_cs+i]+f[14*t_cs+i]+f[15*t_cs+i]+f[16*t_cs+i]+f[17*t_cs+i])*i_rho;
        for(j=0;j<n_cs;j++){
            k=9*(j+2);
            psi[j*t_cs+i]=(f[k*t_cs+i]+f[(k+1)*t_cs+i]+f[(k+2)*t_cs+i]+f[(k+3)*t_cs+i]+f[(k+4)*t_cs+i]+f[(k+5)*t_cs+i]+f[(k+6)*t_cs+i]+f[(k+7)*t_cs+i]+f[(k+8)*t_cs+i])*i_rho;
        }
    }
    for(j=0;j<div;j++){
        // Upper boundary
        en_def=rho[j*y_s]*(T_b-eps[j*y_s]);
        eps[j*y_s]=T_b;
        f[13*t_cs+j*y_s]+=t_3*en_def;            // e_4 direction
        f[16*t_cs+j*y_s]+=i_6*en_def;            // e_7 direction
        f[17*t_cs+j*y_s]+=i_6*en_def;            // e_8 direction
        
        // Lower boundary
        en_def=rho[(j+1)*y_s-1]*(T_a-eps[(j+1)*y_s-1]);
        eps[(j+1)*y_s-1]=T_a;
        f[11*t_cs+(j+1)*y_s-1]+=t_3*en_def;        // e_2 direction
        f[14*t_cs+(j+1)*y_s-1]+=i_6*en_def;        // e_5 direction
        f[15*t_cs+(j+1)*y_s-1]+=i_6*en_def;        // e_6 direction
        
        //        // Supply and extraction at both boundaries
        //        // Species S_i & W_i
        //        for(k=2*n_ss;k<4*n_ss;k++){
        //            // Upper boundary
        //            en_def=rho[j*y_s]*(phi[k]-psi[k*t_cs+j*y_s]);
        //            psi[k*t_cs+j*y_s]=phi[k];
        //            f[(22+k*9)*t_cs+j*y_s]+=t_3*en_def;         // e_4 direction
        //            f[(25+k*9)*t_cs+j*y_s]+=i_6*en_def;         // e_7 direction
        //            f[(26+k*9)*t_cs+j*y_s]+=i_6*en_def;         // e_8 direction
        //
        //            // Lower boundary
        //            en_def=rho[(j+1)*y_s-1]*(phi[k]-psi[k*t_cs+(j+1)*y_s-1]);
        //            psi[k*t_cs+(j+1)*y_s-1]=phi[k];
        //            f[(20+k*9)*t_cs+(j+1)*y_s-1]+=t_3*en_def;   // e_2 direction
        //            f[(23+k*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_5 direction
        //            f[(24+k*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_6 direction
        //        }
        
        // Vertical flow of chemical free energy
        for(k=0;k<n_ss;k++){
            // Lower boundary S_k
            l=2*n_ss+k;
            en_def=rho[(j+1)*y_s-1]*(phi[l]-psi[l*t_cs+(j+1)*y_s-1]);
            psi[l*t_cs+(j+1)*y_s-1]=phi[l];
            f[(20+l*9)*t_cs+(j+1)*y_s-1]+=t_3*en_def;   // e_2 direction
            f[(23+l*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_5 direction
            f[(24+l*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_6 direction
            
            // Upper boundary W_k
            l=3*n_ss+k;
            en_def=rho[j*y_s]*(phi[l]-psi[l*t_cs+j*y_s]);
            psi[l*t_cs+j*y_s]=phi[l];
            f[(22+l*9)*t_cs+j*y_s]+=t_3*en_def;         // e_4 direction
            f[(25+l*9)*t_cs+j*y_s]+=i_6*en_def;         // e_7 direction
            f[(26+l*9)*t_cs+j*y_s]+=i_6*en_def;         // e_8 direction
        }
    }
}

// Now we calculate new momenta and equilibrium distributions at each lattice point
void eq_dists(double *f,double *f_0,double *rho,double *eps,double *psi,double *u,double *u_sq,int t_cs,int *e_is,int n_cs){
    int i,j,k;
    double l_t,e_du;
    
    // Calculate velocity components
    for(i=0;i<t_cs;i++){
        l_t=1/rho[i];
        u[i]=l_t*(f[t_cs+i]-f[3*t_cs+i]+f[5*t_cs+i]-f[6*t_cs+i]-f[7*t_cs+i]+f[8*t_cs+i]);
        u[t_cs+i]=l_t*(f[2*t_cs+i]-f[4*t_cs+i]+f[5*t_cs+i]+f[6*t_cs+i]-f[7*t_cs+i]-f[8*t_cs+i]);
        // Calculate new equilibrium distributions
        u_sq[i]=u[i]*u[i]+u[t_cs+i]*u[t_cs+i];
        l_t=-1.5*u_sq[i];
        f_0[i]=f_9*rho[i]*(1+l_t);									// Fluid (rest particles)
        f_0[9*t_cs+i]=-t_3*rho[i]*eps[i]*u_sq[i];                   // Internal energy (rest particles)
        for(j=0;j<n_cs;j++){
            f_0[(j+2)*9*t_cs+i]=f_9*rho[i]*psi[i+j*t_cs];           // Chemical species (rest particles)
        }
        for(k=1;k<5;k++){
            e_du=e_is[2*k]*u[i]+e_is[2*k+1]*u[t_cs+i];
            f_0[k*t_cs+i]=i_9*rho[i]*(1+3*e_du+4.5*e_du*e_du+l_t);									// Fluid (particle index k)
            f_0[(k+9)*t_cs+i]=i_9*rho[i]*eps[i]*(1.5+1.5*e_du+4.5*e_du*e_du+l_t);					// Internal energy (particle index k)
            for(j=0;j<n_cs;j++){
                f_0[(k+9*(j+2))*t_cs+i]=i_9*rho[i]*psi[i+j*t_cs]*(1+3*e_du);                        // Chemical species (particle index k)
            }
            e_du=e_is[2*(k+4)]*u[i]+e_is[2*(k+4)+1]*u[t_cs+i];
            f_0[(k+4)*t_cs+i]=i_36*rho[i]*(1+3*e_du+4.5*e_du*e_du+l_t);								// Fluid particles (particle index k+4)
            f_0[(k+13)*t_cs+i]=i_36*rho[i]*eps[i]*(3+6*e_du+4.5*e_du*e_du+l_t);						// Internal energy (particle index k+4)
            for(j=0;j<n_cs;j++){
                f_0[(k+9*(j+2)+4)*t_cs+i]=i_36*rho[i]*psi[i+j*t_cs]*(1+3*e_du);                     // Chemical species (particle index k+4)
            }
        }
    }
}

// Apply collision step
void coll_step(double *f,double *f_0,double *u,double *eps,double *psi,double T_0,double i_T_0,int *e_is,double b_g0,
               double i_tau_v,double i_tau_c,double *i_tau_s,double c,int t_cs,double *omega,double *E_f,
               double *del_H_r,double *A_f,int *stoi,int n_cs,int n_ss,int R){
    int i,j,k,l,m,n,ind;
    double r_F;
    
    for(i=0;i<t_cs;i++){
        for(k=0;k<9;k++){
            // Fluid collisions and buoyancy force
            f[k*t_cs+i]+=f_0[k*t_cs+i]*(i_tau_v+b_g0*i_T_0*(eps[i]-T_0)*c*(e_is[2*k+1]-u[t_cs+i]))-i_tau_v*f[k*t_cs+i];   // Convection on
            //            f[k*t_cs+i]+=i_tau_v*(f_0[k*t_cs+i]-f[k*t_cs+i]);        // Convection off
            // Internal energy collisions
            f[(k+9)*t_cs+i]+=i_tau_c*(f_0[(k+9)*t_cs+i]-f[(k+9)*t_cs+i]);
            
            // Chemical species collision
            for(j=0;j<n_cs;j++){
                ind=(k+(j+2)*9)*t_cs+i;
                f[ind]+=i_tau_s[j]*(f_0[ind]-f[ind]);
            }
            
            // Reactions
            for(j=0;j<R;j++){
                r_F=A_f[j]*exp(-E_f[j]/eps[i])*omega[k];
                for(l=0;l<n_cs;l++){
                    r_F*=pow(psi[l*t_cs+i],stoi[j*n_cs+l]);
                }
                // Enthalpy change
                f[(k+9)*t_cs+i]-=del_H_r[j]*r_F;
                // Chemical species changes
                for(l=0;l<n_cs;l++){
                    f[(k+(l+2)*9)*t_cs+i]-=r_F*(stoi[j*n_cs+l]-stoi[n_cs*R+j*n_cs+l]);
                }
            }
        }
    }
}

int main(int argc, char **argv){
    // Declare constants etc.
    int i,j,k,t,m,num_procs,rank,div,t_cs,left,right,fc_s=0,grad=0,n_ss=1,c_m=0,n_cs=4*n_ss+c_m,R=6*n_ss,*stoi;
    int t_end=5e5;
    int t_int=t_end/500;
    int x_s=400,y_s=x_s/2;
    int l_1=(y_s-1)*sizeof(double),l_2,l_3,e_is[18]={0,0,1,0,0,1,-1,0,0,-1,1,1,-1,1,-1,-1,1,-1};
    double i_nds=1.0/(y_s*x_s),fd=0.03,kl=0.061,Ra=1e4,Pr=0.71,T_s,x_fac,fac=0.1;
    double T_0=1.5,T_a=T_0,T_b=T_0,i_T_0=1/T_0,c=sqrt(3*T_0),S_B=1e2,per=10e4;
    T_a=2; T_b=1;
    double H=y_s*c,E_f[R],A_f[R],del_H_r[R],tau_v=0.6,tau_c=0.5*((tau_v-0.5)/Pr+1),tau_s[n_cs],i_tau_s[n_cs],phi[n_cs];
    double nu=i_3*(tau_v-0.5)*c*c,chi=t_3*(tau_c-0.5)*c*c,i_tau_v=1/tau_v,i_tau_c=1/tau_c,b_g0=Ra*chi*nu/pow(H,3);
    double F=Ra*chi*chi*nu/(b_g0*pow(H,4));
    double *rho,*eps,*psi,*u,*u_sq,*f,*f_0,*u_x_all,*u_y_all,*e_all,*psi_all,omega[9]={f_9,i_9,i_9,i_9,i_9,i_36,i_36,i_36,i_36};
    char d_out[16],run_st[2];
    FILE *fout_1;
    
    MPI_Status status; MPI_Comm Comm_cart; MPI_Init(&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD,&num_procs); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    div=x_s/num_procs; t_cs=div*y_s;
    seed=-(signed)time(NULL)+pow(rank,2.7);
    l_2=(div-1)*y_s*sizeof(double);
    l_3=(n_cs+2)*9*t_cs*sizeof(double);
    T_s=rank*3.0/num_procs;
    x_fac=3.0/(x_s-1);
    
    left=rank-1; right=rank+1;
    if(rank==0){left=num_procs-1;}
    else if(rank==num_procs-1){right=0;}
    
    // Assign memory for arrays
    rho=malloc(t_cs*sizeof(double)); eps=malloc(t_cs*sizeof(double));
    psi=malloc(n_cs*t_cs*sizeof(double)); psi_all=malloc(n_cs*y_s*x_s*sizeof(double));
    e_all=malloc(y_s*x_s*sizeof(double));
    u=malloc(2*t_cs*sizeof(double)); u_sq=malloc(t_cs*sizeof(double));
    u_x_all=malloc(y_s*x_s*sizeof(double)); u_y_all=malloc(y_s*x_s*sizeof(double));
    f=malloc((n_cs+2)*9*t_cs*sizeof(double)); f_0=malloc((n_cs+2)*9*t_cs*sizeof(double));
    stoi=malloc(2*n_cs*R*sizeof(int)); memset(stoi,0,2*n_cs*R*sizeof(int));
    
    //T_a=T_0-sin(-2.0*pi/per);
    //T_b=T_0-sin(-2.0*pi/per);
    
    for(i=0;i<n_ss;i++){
        // Stoichiometry
        stoi[i*6*n_cs+2*n_ss+i]=1;              // First row reactants: 1 molecule of S_i in
        stoi[n_cs*R+i*6*n_cs+i]=1;              // First row products: 1 molecule of A_i out
        stoi[(i*6+1)*n_cs+i]=1;                 // Second row reactants: 1 molecule of A_i in
        stoi[n_cs*R+(i*6+1)*n_cs+2*n_ss+i]=1;   // Second row products: 1 molecule of S_i out
        stoi[(i*6+2)*n_cs+i]=1;                 // Third row reactants: 1 molecule of A_i in
        stoi[(i*6+2)*n_cs+n_ss+i]=2;            // Third row reactants: 2 molecules of B_i in
        stoi[n_cs*R+(i*6+2)*n_cs+n_ss+i]=3;     // Third row products: 3 molecules of B_i out
        stoi[(i*6+3)*n_cs+n_ss+i]=3;            // Fourth row reactants: 3 molecules of B_i in
        stoi[n_cs*R+(i*6+3)*n_cs+i]=1;          // Fourth row products: 1 molecule of A_i out
        stoi[n_cs*R+(i*6+3)*n_cs+n_ss+i]=2;     // Fourth row products: 2 molecules of B_i out
        stoi[(i*6+4)*n_cs+n_ss+i]=1;            // Fifth row reactants: 1 molecule of B_i in
        stoi[n_cs*R+(i*6+4)*n_cs+3*n_ss+i]=1;   // Fifth row products: 1 molecule of W_i out
        stoi[(i*6+5)*n_cs+3*n_ss+i]=1;          // Sixth row reactants: 1 molecule of W_i in
        stoi[n_cs*R+(i*6+5)*n_cs+n_ss+i]=1;     // Sixth row products: 1 molecules of B_i out
        
        // Diffusion coefficients
        tau_s[i]=0.6;                           // A_i
        tau_s[n_ss+i]=0.5*(tau_s[i]+0.5);       // B_i
        tau_s[2*n_ss+i]=3;                      // S_i
        tau_s[3*n_ss+i]=3;                      // W_i
        for(j=0;j<4;j++){
            i_tau_s[j*n_ss+i]=1/tau_s[j*n_ss+i];
        }
        
        // Reaction rate parameters
        // S_i->A_i
        phi[2*n_ss+i]=S_B;                      // Boundary concentration for species S_i
        E_f[i*6]=1e-3*T_0;
        A_f[i*6]=fac*fd*exp(E_f[i*6]/T_0)/S_B;
        // A_i->S_i
        E_f[i*6+1]=E_f[i*6];
        A_f[i*6+1]=fac*fd*exp(E_f[i*6+1]/T_0);
        
        // A_i+2B_i->3B_i
        E_f[i*6+2]=1.5;
        A_f[i*6+2]=fac*exp(E_f[i*6+2]/T_0);
        
        // 3B_i->A_i+2B_i
        A_f[i*6+3]=fac*1e-5;
        
        // Delta H=0.025 (endothermic)
        E_f[i*6+3]=E_f[i*6+2];
        
        //        // Spot species 0
        //        if(i==0){
        //            // Exothermic
        //            E_f[i*6+3]=E_f[i*6+2]+0.2;
        //        }
        //        // Spot species 1
        //        else if(i==1){
        //            // 3B_i->A_i+2B_i
        //            // Endothermic
        //            E_f[i*6+3]=E_f[i*6+2]-0.25;
        //        }
        //        // Species 2+
        //        else{
        //            E_f[i*6+3]=E_f[i*6+2];
        //        }
        
        // B_i->W_i
        phi[3*n_ss+i]=0;                        // Boundary concentration for species W_i
        E_f[i*6+4]=1e-3*T_0;
        A_f[i*6+4]=fac*(fd+kl)*exp(E_f[i*6+4]/T_0);
        // W_i->B_i
        E_f[i*6+5]=E_f[i*6+4];
        A_f[i*6+5]=fac*1e-5;
        
        // Reaction enthalpies
        for(j=0;j<3;j++){
            del_H_r[i*6+2*j]=E_f[i*6+2*j]-E_f[i*6+2*j+1];
            del_H_r[i*6+2*j+1]=-del_H_r[i*6+2*j];
        }
    }
    
    if(rank==0){
        //            sprintf(run_st,"%d",m); strcpy(d_out,"opts_reac_");
        //            strcat(d_out,run_st); strcat(d_out,".txt");
        //            fout_1=fopen(d_out,"w");
        fout_1=fopen("opts_reac_1.txt","w");
        fprintf(fout_1,"%d %d %d %d %d %d %d %d %6.4f %6.4f %d ",y_s,x_s,t_end,t_int,0,0,0,n_cs,tau_v,tau_c,1);
    }
    
    // Set uniform initial density, random temperature of mean T_0 and 0 velocity
    for(i=0;i<t_cs;i++){
        if(grad==1){
            eps[i]=(T_s+floor(i/y_s)*x_fac)*(1+0.1*2.0*(ran2(&seed)-0.5));
        }
        else{
            eps[i]=T_0*(1+0.2*2.0*(ran2(&seed)-0.5));
        }
        //eps[i]=T_0;
        
        for(j=0;j<n_ss;j++){
            psi[j*t_cs+i]=1-0.01*ran2(&seed);   // Species A_j
            if(ran2(&seed)<0.05){               // Species B_j
                psi[(n_ss+j)*t_cs+i]=1;
            }
            else{
                psi[(n_ss+j)*t_cs+i]=0;
            }
            
            //psi[j*t_cs+i]=1;                 // Species A_j
            //psi[(n_ss+j)*t_cs+i]=1;          // Species B_j
            
            psi[(2*n_ss+j)*t_cs+i]=phi[2*n_ss+j];   // Species S_j
            psi[(3*n_ss+j)*t_cs+i]=phi[3*n_ss+j];   // Species W_j
        }
        
        // Set initial distributions, assume equilibrium to start with
        f[i]=f_9;                                   // Fluid (rest particles)
        // Internal energy and chemical species
        for(j=1;j<n_cs+2;j++){
            f[9*j*t_cs+i]=0;
        }
        for(k=1;k<5;k++){
            f[k*t_cs+i]=i_9;                                // Fluid (particle index k)
            f[(k+4)*t_cs+i]=i_36;                           // Fluid (particle index k+4)
            f[(k+9)*t_cs+i]=i_6*eps[i];                     // Internal energy (particle index k)
            f[(k+13)*t_cs+i]=i_12*eps[i];                   // Internal energy (particle index k+4)
            for(j=0;j<n_cs;j++){
                f[(9*(j+2)+k)*t_cs+i]=i_6*psi[j*t_cs+i];	// Chemical species (particle index k)
                f[(9*(j+2)+k+4)*t_cs+i]=i_12*psi[j*t_cs+i];	// Chemical species (particle index k+4)
            }
        }
    }
    
    // Run simulation
    for(t=0;t<=t_end;t++){
        // Make temporary copies of all fields
        memcpy(&f_0[0],&f[0],l_3);
        // Streaming step
        for(k=0;k<(2+n_cs)*9;k+=9){
            // Exchange leftward moving distributions
            MPI_Sendrecv(&f_0[(k+3)*t_cs],y_s,MPI_DOUBLE,left,3+10*k,&f[(k+4)*t_cs-y_s],y_s,MPI_DOUBLE,right,3+10*k,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_0[(k+6)*t_cs],y_s,MPI_DOUBLE,left,6+20*k,&f[(k+7)*t_cs-y_s],y_s,MPI_DOUBLE,right,6+20*k,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_0[(k+7)*t_cs],y_s,MPI_DOUBLE,left,7+30*k,&f[(k+8)*t_cs-y_s],y_s,MPI_DOUBLE,right,7+30*k,MPI_COMM_WORLD,&status);
            // Exchange rightward moving distributions
            MPI_Sendrecv(&f_0[(k+2)*t_cs-y_s],y_s,MPI_DOUBLE,right,1+10*k,&f[(k+1)*t_cs],y_s,MPI_DOUBLE,left,1+10*k,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_0[(k+6)*t_cs-y_s],y_s,MPI_DOUBLE,right,5+20*k,&f[(k+5)*t_cs],y_s,MPI_DOUBLE,left,5+20*k,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_0[(k+9)*t_cs-y_s],y_s,MPI_DOUBLE,right,8+30*k,&f[(k+8)*t_cs],y_s,MPI_DOUBLE,left,8+30*k,MPI_COMM_WORLD,&status);
            //            // Enforce bounce back on the left and right boundaries
            //            if(rank==0){
            //                memcpy(&f[(k+1)*t_cs],&f_0[(k+3)*t_cs],y_s*sizeof(double));
            //                memcpy(&f[(k+5)*t_cs],&f_0[(k+7)*t_cs],y_s*sizeof(double));
            //                memcpy(&f[(k+8)*t_cs],&f_0[(k+6)*t_cs],y_s*sizeof(double));
            //            }
            //            else if(rank==num_procs-1){
            //                memcpy(&f[(k+4)*t_cs-y_s],&f_0[(k+2)*t_cs-y_s],y_s*sizeof(double));
            //                memcpy(&f[(k+8)*t_cs-y_s],&f_0[(k+6)*t_cs-y_s],y_s*sizeof(double));
            //                memcpy(&f[(k+7)*t_cs-y_s],&f_0[(k+9)*t_cs-y_s],y_s*sizeof(double));
            //            }
        }
        
        bb_per_bc(f,f_0,t_cs,div,y_s,l_1,l_2,n_cs);                             // Streaming and bounce back on boundaries
        calc_dens(f,rho,eps,psi,T_a,T_b,phi,y_s,div,n_cs,t_cs,fc_s,grad,        // Calculate all new densities
                  T_s,F,x_fac,n_ss,rank,num_procs,x_s,omega);
        eq_dists(f,f_0,rho,eps,psi,u,u_sq,t_cs,e_is,n_cs);                      // Calculate new equilibrium distributions
        coll_step(f,f_0,u,eps,psi,T_0,i_T_0,e_is,b_g0,i_tau_v,i_tau_c,i_tau_s,  // Collision step
                  c,t_cs,omega,E_f,del_H_r,A_f,stoi,n_cs,n_ss,R);
        
        // Adjust temperatures
        //T_a=T_0-sin(2.0*pi*t/per);
        //T_b=T_0-sin(2.0*pi*t/per);
        
        // Gather and print whole array
        if(t%t_int==0){
            MPI_Gather(u,t_cs,MPI_DOUBLE,u_x_all,t_cs,MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Gather(&u[t_cs],t_cs,MPI_DOUBLE,u_y_all,t_cs,MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Gather(eps,t_cs,MPI_DOUBLE,e_all,t_cs,MPI_DOUBLE,0,MPI_COMM_WORLD);
            for(j=0;j<n_cs;j++){
                MPI_Gather(&psi[j*t_cs],t_cs,MPI_DOUBLE,&psi_all[j*y_s*x_s],t_cs,MPI_DOUBLE,0,MPI_COMM_WORLD);
            }
            if(rank==0){
                for(i=0;i<y_s*x_s;i++){
                    fprintf(fout_1,"%10.8f ",e_all[i]);
                }
                for(i=0;i<y_s*x_s;i++){
                    fprintf(fout_1,"%10.8f ",c*u_x_all[i]);
                }
                for(i=0;i<y_s*x_s;i++){
                    fprintf(fout_1,"%10.8f ",c*u_y_all[i]);
                }
                for(j=0;j<n_cs;j++){
                    for(i=0;i<y_s*x_s;i++){
                        fprintf(fout_1,"%10.8f ",psi_all[j*y_s*x_s+i]);
                    }
                }
            }
        }
    }
    if(rank==0){
        fclose(fout_1);
    }
    MPI_Finalize();
}
