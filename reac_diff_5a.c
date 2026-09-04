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
void bb_per_bc(double *f,double *f_0,int t_cs,int div,int y_s,size_t l_1,size_t l_2,int n_cs){
    int j,k;
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
               int div,int n_cs,int t_cs,int n_ss){
    int i,j,k;
    // int l;
    double en_def,i_rho;
    
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
        // Constant temperature boundaries
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
        
        // Supply and extraction at both boundaries, keep both boundaries at the respective phi conc's
        // Species S_i & W_i
        for(k=2*n_ss;k<4*n_ss;k++){
            // Upper boundary
            en_def=rho[j*y_s]*(phi[k]-psi[k*t_cs+j*y_s]);
            psi[k*t_cs+j*y_s]=phi[k];
            f[(22+k*9)*t_cs+j*y_s]+=t_3*en_def;         // e_4 direction
            f[(25+k*9)*t_cs+j*y_s]+=i_6*en_def;         // e_7 direction
            f[(26+k*9)*t_cs+j*y_s]+=i_6*en_def;         // e_8 direction

            // Lower boundary
            en_def=rho[(j+1)*y_s-1]*(phi[k]-psi[k*t_cs+(j+1)*y_s-1]);
            psi[k*t_cs+(j+1)*y_s-1]=phi[k];
            f[(20+k*9)*t_cs+(j+1)*y_s-1]+=t_3*en_def;   // e_2 direction
            f[(23+k*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_5 direction
            f[(24+k*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_6 direction
        }
        
        // // Vertical flow of chemical free energy, supply S at lower boundary and extract W at upper boundary
        // for(k=0;k<n_ss;k++){
        //     // Upper boundary W_k
        //     l=3*n_ss+k;
        //     en_def=rho[j*y_s]*(phi[l]-psi[l*t_cs+j*y_s]);
        //     psi[l*t_cs+j*y_s]=phi[l];
        //     f[(22+l*9)*t_cs+j*y_s]+=t_3*en_def;         // e_4 direction
        //     f[(25+l*9)*t_cs+j*y_s]+=i_6*en_def;         // e_7 direction
        //     f[(26+l*9)*t_cs+j*y_s]+=i_6*en_def;         // e_8 direction
             
        //     // Lower boundary S_k
        //     l=2*n_ss+k;
        //     en_def=rho[(j+1)*y_s-1]*(phi[l]-psi[l*t_cs+(j+1)*y_s-1]);
        //     psi[l*t_cs+(j+1)*y_s-1]=phi[l];
        //     f[(20+l*9)*t_cs+(j+1)*y_s-1]+=t_3*en_def;   // e_2 direction
        //     f[(23+l*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_5 direction
        //     f[(24+l*9)*t_cs+(j+1)*y_s-1]+=i_6*en_def;   // e_6 direction
        // }
    }
}

// Now we calculate new momenta and equilibrium distributions at each lattice point
void eq_dists(double *f,double *f_0,double *rho,double *eps,double *psi,double *u,int t_cs,int *e_is,int n_cs){
    int i,j,k;
    double l_t,e_du,u_sq;
    
    // Calculate velocity components
    for(i=0;i<t_cs;i++){
        l_t=1/rho[i];
        u[i]=l_t*(f[t_cs+i]-f[3*t_cs+i]+f[5*t_cs+i]-f[6*t_cs+i]-f[7*t_cs+i]+f[8*t_cs+i]);
        u[t_cs+i]=l_t*(f[2*t_cs+i]-f[4*t_cs+i]+f[5*t_cs+i]+f[6*t_cs+i]-f[7*t_cs+i]-f[8*t_cs+i]);
        // Calculate new equilibrium distributions
        u_sq=u[i]*u[i]+u[t_cs+i]*u[t_cs+i];
        l_t=-1.5*u_sq;
        f_0[i]=f_9*rho[i]*(1+l_t);							// Fluid (rest particles)
        f_0[9*t_cs+i]=-t_3*rho[i]*eps[i]*u_sq;              // Internal energy (rest particles)
        for(j=0;j<n_cs;j++){
            f_0[(j+2)*9*t_cs+i]=f_9*rho[i]*psi[i+j*t_cs];   // Chemical species (rest particles)
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
               double *del_H_r,double *A_f,int *stoi,int n_cs,int R){
    int i,j,k,l;
    double r_F;
    
    for(i=0;i<t_cs;i++){
        double rate[R];
		for(j=0;j<R;j++){
			rate[j]=A_f[j]*exp(-E_f[j]/eps[i]);
			for(l=0;l<n_cs;l++){
				int s=stoi[j*n_cs+l];
				double p=psi[l*t_cs+i],q=1.0;
				while(s--) q*=p;
				rate[j]*=q;
			}
		}

        for(k=0;k<9;k++){
			// Fluid collisions and buoyancy force
            f[k*t_cs+i]+=f_0[k*t_cs+i]*(i_tau_v+b_g0*i_T_0*(eps[i]-T_0)*c*(e_is[2*k+1]-u[t_cs+i]))-i_tau_v*f[k*t_cs+i];   // Convection on
            // f[k*t_cs+i]+=i_tau_v*(f_0[k*t_cs+i]-f[k*t_cs+i]);        // Convection off
            // Internal energy collisions
			f[(k+9)*t_cs+i]+=i_tau_c*(f_0[(k+9)*t_cs+i]-f[(k+9)*t_cs+i]);
			// Chemical species collision
			for(j=0;j<n_cs;j++){
				int ind=(k+(j+2)*9)*t_cs+i;
				f[ind]+=i_tau_s[j]*(f_0[ind]-f[ind]);
			}
			// Reactions
			for(j=0;j<R;j++){
				r_F=rate[j]*omega[k];
				f[(k+9)*t_cs+i]-=del_H_r[j]*r_F;
				for(l=0;l<n_cs;l++){
					f[(k+(l+2)*9)*t_cs+i]-=r_F*(stoi[j*n_cs+l]-stoi[n_cs*R+j*n_cs+l]);
				}
			}
		}
    }
}

int main(int argc, char **argv){
    // Declare constants etc.
    int i,j,k,t,m,num_procs,rank,div,t_cs,left,right;
    int n_ss=2,n_cs=4*n_ss,R=6*n_ss,*stoi;
    n_ss=2;
    int t_end=10e4;
    int t_int=t_end/200;
    int x_s=256,y_s=x_s/2;
    int e_is[18]={0,0,1,0,0,1,-1,0,0,-1,1,1,-1,1,-1,-1,1,-1};
    double fd=0.03,kl=0.061,Pr=0.71,fac=0.1;
    double Ra=1e4;
    double T_0=1.0,T_a=T_0,T_b=T_0,i_T_0=1/T_0,c=sqrt(3*T_0),S_B=1e2;
    fac=0.1;
    T_a=1.5; T_b=0.5;
    double H=y_s*c,E_f[R],A_f[R],del_H_r[R],tau_v=0.58,tau_c=0.5*((tau_v-0.5)/Pr+1),tau_s[n_cs],i_tau_s[n_cs],phi[n_cs];
    double nu=i_3*(tau_v-0.5)*c*c,chi=t_3*(tau_c-0.5)*c*c,i_tau_v=1/tau_v,i_tau_c=1/tau_c,b_g0;
    if(T_a==T_b){
        b_g0=0;
    }
    else{
        b_g0=Ra*chi*nu/(pow(H,3)*(T_a-T_b));
    }
    // double F=Ra*chi*chi*nu/(b_g0*pow(H,4));
    double *rho,*eps,*psi,*u,*f_a,*f_b,*f_t,*u_x_all,*u_y_all,*e_all,*psi_all,omega[9]={f_9,i_9,i_9,i_9,i_9,i_36,i_36,i_36,i_36};
    FILE *fout_1;
    
    MPI_Status status; 
    // MPI_Comm Comm_cart; 
    MPI_Init(&argc,&argv); 
    MPI_Comm_size(MPI_COMM_WORLD,&num_procs); 
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    div=x_s/num_procs; t_cs=div*y_s;
    // Check the decomposition is valid: x_s must divide evenly and each block needs >=2 columns
	if(x_s%num_procs!=0 || div<2){
		if(rank==0){
			fprintf(stderr,"Invalid decomposition: x_s=%d must be divisible by num_procs=%d with at least 2 columns per rank (got div=%d)\n",x_s,num_procs,div);
		}
		MPI_Abort(MPI_COMM_WORLD,1);
	}

	left=rank-1;
	right=rank+1;
	if(rank==0){
		left=num_procs-1;
	}
	if(rank==num_procs-1){
		right=0;
	}

    seed = -(long)(time(NULL) + 7919L*rank);	// Seed random number generator
	// seed = -12345L - 7919L*rank;				// Seed random number generator (fixed for reproducibility)
	
    size_t l_1=(y_s-1)*sizeof(double),l_2,l_3;
	l_2=(size_t)(div-1)*y_s*sizeof(double);
	l_3=(size_t)(n_cs+2)*9*t_cs*sizeof(double);

    // Assign memory for arrays
    rho=malloc(t_cs*sizeof(double)); 
    eps=malloc(t_cs*sizeof(double));
    psi=malloc(n_cs*t_cs*sizeof(double)); 
    u=malloc(2*t_cs*sizeof(double));
    f_a=malloc((n_cs+2)*9*t_cs*sizeof(double)); f_b=malloc((n_cs+2)*9*t_cs*sizeof(double));
    stoi=malloc(2*n_cs*R*sizeof(int)); memset(stoi,0,2*n_cs*R*sizeof(int));

    psi_all=NULL; e_all=NULL; u_x_all=NULL; u_y_all=NULL; fout_1=NULL;
	if(rank==0){
        psi_all=malloc(n_cs*y_s*x_s*sizeof(double));
        e_all=malloc(y_s*x_s*sizeof(double));
        u_x_all=malloc(y_s*x_s*sizeof(double)); 
        u_y_all=malloc(y_s*x_s*sizeof(double));
    }
	
	if(eps==NULL||psi==NULL||u==NULL||f_a==NULL||f_b==NULL||stoi==NULL||(rank==0&&(psi_all==NULL||e_all==NULL||u_x_all==NULL||u_y_all==NULL))){
		fprintf(stderr,"Rank %d: allocation failed\n",rank);
		MPI_Abort(MPI_COMM_WORLD,2);
	}

    memset(f_b,0,l_3);

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
        
        // Boundary concentration for species S_i
        phi[2*n_ss+i]=S_B;
        // Boundary concentration for species W_i
        phi[3*n_ss+i]=0;
        
        // Reaction rate parameters
        // S_i->A_i
        E_f[i*6]=1e-3*T_0;
        A_f[i*6]=fac*fd*exp(E_f[i*6]/T_0)/S_B;
        // A_i->S_i
        E_f[i*6+1]=E_f[i*6];
        A_f[i*6+1]=fac*fd*exp(E_f[i*6+1]/T_0);
        
        // A_i+2B_i->3B_i
        E_f[i*6+2]=1.0;
        A_f[i*6+2]=fac*exp(E_f[i*6+2]/T_0);
        
        // 3B_i->A_i+2B_i
        E_f[i*6+3]=E_f[i*6+2];
        A_f[i*6+3]=fac*1e-5;
        
        // Spot species 0
        if(i==0){
            // Exothermic
            E_f[i*6+3]=E_f[i*6+2]+0.02;
        }
        // Spot species 1
        else if(i==1){
            // 3B_i->A_i+2B_i
            // Endothermic
            E_f[i*6+3]=E_f[i*6+2]-0.025;
        }
        // // Species 2+
        // else{
        //     E_f[i*6+3]=E_f[i*6+2];
        // }
        
        // B_i->W_i
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
		fout_1=fopen("opts_reac_1.txt","w");
		if(fout_1==NULL){
			fprintf(stderr,"Rank 0: could not open output file\n");
			MPI_Abort(MPI_COMM_WORLD,4);
		}
		fprintf(fout_1,"%d %d %d %d %d ",y_s,x_s,t_end,t_int,n_cs);
	}

    // Scatter a few trigger patches per rank, independently for each spot species
	int n_patch=3,pr=4;
	int px[n_ss][n_patch],py[n_ss][n_patch];
	for(j=0;j<n_ss;j++){
		for(m=0;m<n_patch;m++){
			px[j][m]=(int)(ran2(&seed)*div);
			py[j][m]=pr+(int)(ran2(&seed)*(y_s-2*pr));
		}
	}

    // Set uniform initial density, random temperature of mean T_0 and 0 velocity
    for(i=0;i<t_cs;i++){
        eps[i]=T_0*(1+0.2*2.0*(ran2(&seed)-0.5));

        int jl=i/y_s,il=i%y_s;					// local column, row
        for(j=0;j<n_ss;j++){
            int in_patch=0;
			for(m=0;m<n_patch;m++){
				int dx=jl-px[j][m],dy=il-py[j][m];
				if(dx*dx+dy*dy<pr*pr) in_patch=1;
			}
            if(in_patch){
                psi[j*t_cs+i]        =0.50*(1+0.02*2.0*(ran2(&seed)-0.5));	// A
                psi[(n_ss+j)*t_cs+i] =0.25*(1+0.02*2.0*(ran2(&seed)-0.5));	// B
            }
            else{
                psi[j*t_cs+i]        =1.0;
                psi[(n_ss+j)*t_cs+i] =0.0;
            }
            psi[(2*n_ss+j)*t_cs+i]=phi[2*n_ss+j];
            psi[(3*n_ss+j)*t_cs+i]=phi[3*n_ss+j];
        }
        
        // Set initial distributions, assume equilibrium to start with
        f_a[i]=f_9;                                             // Fluid (rest particles)
        // Internal energy and chemical species
        for(j=1;j<n_cs+2;j++){
            f_a[9*j*t_cs+i]=0;
        }
        for(k=1;k<5;k++){
            f_a[k*t_cs+i]=i_9;                                  // Fluid (particle index k)
            f_a[(k+4)*t_cs+i]=i_36;                             // Fluid (particle index k+4)
            f_a[(k+9)*t_cs+i]=i_6*eps[i];                       // Internal energy (particle index k)
            f_a[(k+13)*t_cs+i]=i_12*eps[i];                     // Internal energy (particle index k+4)
            for(j=0;j<n_cs;j++){
                f_a[(9*(j+2)+k)*t_cs+i]=i_6*psi[j*t_cs+i];	    // Chemical species (particle index k)
                f_a[(9*(j+2)+k+4)*t_cs+i]=i_12*psi[j*t_cs+i];	// Chemical species (particle index k+4)
            }
        }
    }
    
    // Run simulation
    for(t=0;t<=t_end;t++){
        // Streaming step
        for(k=0;k<(2+n_cs)*9;k+=9){
            int tg=k*10;
            // Exchange leftward moving distributions
            MPI_Sendrecv(&f_a[(k+3)*t_cs],y_s,MPI_DOUBLE,left,3+tg,&f_b[(k+4)*t_cs-y_s],y_s,MPI_DOUBLE,right,3+tg,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_a[(k+6)*t_cs],y_s,MPI_DOUBLE,left,6+tg,&f_b[(k+7)*t_cs-y_s],y_s,MPI_DOUBLE,right,6+tg,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_a[(k+7)*t_cs],y_s,MPI_DOUBLE,left,7+tg,&f_b[(k+8)*t_cs-y_s],y_s,MPI_DOUBLE,right,7+tg,MPI_COMM_WORLD,&status);
            // Exchange rightward moving distributions
            MPI_Sendrecv(&f_a[(k+2)*t_cs-y_s],y_s,MPI_DOUBLE,right,1+tg,&f_b[(k+1)*t_cs],y_s,MPI_DOUBLE,left,1+tg,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_a[(k+6)*t_cs-y_s],y_s,MPI_DOUBLE,right,5+tg,&f_b[(k+5)*t_cs],y_s,MPI_DOUBLE,left,5+tg,MPI_COMM_WORLD,&status);
            MPI_Sendrecv(&f_a[(k+9)*t_cs-y_s],y_s,MPI_DOUBLE,right,8+tg,&f_b[(k+8)*t_cs],y_s,MPI_DOUBLE,left,8+tg,MPI_COMM_WORLD,&status);
            //            // Enforce bounce back on the left and right boundaries
            //            if(rank==0){
            //                memcpy(&f_b[(k+1)*t_cs],&f_a[(k+3)*t_cs],y_s*sizeof(double));
            //                memcpy(&f_b[(k+5)*t_cs],&f_a[(k+7)*t_cs],y_s*sizeof(double));
            //                memcpy(&f_b[(k+8)*t_cs],&f_a[(k+6)*t_cs],y_s*sizeof(double));
            //            }
            //            else if(rank==num_procs-1){
            //                memcpy(&f_b[(k+4)*t_cs-y_s],&f_a[(k+2)*t_cs-y_s],y_s*sizeof(double));
            //                memcpy(&f_b[(k+8)*t_cs-y_s],&f_a[(k+6)*t_cs-y_s],y_s*sizeof(double));
            //                memcpy(&f_b[(k+7)*t_cs-y_s],&f_a[(k+9)*t_cs-y_s],y_s*sizeof(double));
            //            }
        }
        
        bb_per_bc(f_b,f_a,t_cs,div,y_s,l_1,l_2,n_cs);                             // Streaming and bounce back on boundaries
        
        // Rest populations do not stream, and bb_per_bc never writes them
		for(k=0;k<(2+n_cs)*9;k+=9){
            memcpy(&f_b[k*t_cs],&f_a[k*t_cs],t_cs*sizeof(double));
        }

        f_t=f_a; f_a=f_b; f_b=f_t;
		
        calc_dens(f_a,rho,eps,psi,T_a,T_b,phi,y_s,div,n_cs,t_cs,n_ss);       // Calculate all new densities
        eq_dists(f_a,f_b,rho,eps,psi,u,t_cs,e_is,n_cs);                      // Calculate new equilibrium distributions
        coll_step(f_a,f_b,u,eps,psi,T_0,i_T_0,e_is,b_g0,i_tau_v,i_tau_c,i_tau_s,  // Collision step
                  c,t_cs,omega,E_f,del_H_r,A_f,stoi,n_cs,R);
        
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
                    fprintf(fout_1,"%8.6f ",e_all[i]);
                }
                for(i=0;i<y_s*x_s;i++){
                    fprintf(fout_1,"%8.6f ",c*u_x_all[i]);
                }
                for(i=0;i<y_s*x_s;i++){
                    fprintf(fout_1,"%8.6f ",c*u_y_all[i]);
                }
                for(j=0;j<n_cs;j++){
                    for(i=0;i<y_s*x_s;i++){
                        fprintf(fout_1,"%8.6f ",psi_all[j*y_s*x_s+i]);
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
