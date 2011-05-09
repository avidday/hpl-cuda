/* 
 *
 * -- High Performance Computing Linpack Benchmark for CUDA
 *    hpl-cuda - 0.001 - 2010
 *
 *    David Martin (cuda@avidday.net)
 *    (C) Copyright 2010-2011 All Rights Reserved
 *
 *    portions of this code written by
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 
/*
 * Include files
 */
#include <cublas.h>
#include "hpl.h"
#include "hpl_gpusupport.h"

    
#ifdef STDC_HEADERS
void HPL_pdupdateTT
(
   HPL_T_panel *                    PBCST,
   int *                            IFLAG,
   HPL_T_panel *                    PANEL,
   const int                        NN
)
#else
void HPL_pdupdateTT
( PBCST, IFLAG, PANEL, NN )
   HPL_T_panel *                    PBCST;
   int *                            IFLAG;
   HPL_T_panel *                    PANEL;
   const int                        NN;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdupdateTT broadcast - forward the panel PBCST and simultaneously
 * applies the row interchanges and updates part of the trailing  (using
 * the panel PANEL) submatrix.
 *
 * Arguments
 * =========
 *
 * PBCST   (local input/output)          HPL_T_panel *
 *         On entry,  PBCST  points to the data structure containing the
 *         panel (to be broadcast) information.
 *
 * IFLAG   (local output)                int *
 *         On exit,  IFLAG  indicates  whether or not  the broadcast has
 *         been completed when PBCST is not NULL on entry. In that case,
 *         IFLAG is left unchanged.
 *
 * PANEL   (local input/output)          HPL_T_panel *
 *         On entry,  PANEL  points to the data structure containing the
 *         panel (to be updated) information.
 *
 * NN      (local input)                 const int
 *         On entry, NN specifies  the  local  number  of columns of the
 *         trailing  submatrix  to be updated  starting  at the  current
 *         position. NN must be at least zero.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   double                    * Aptr, * L1ptr, * L2ptr, * Uptr, * dpiv;
   int                       * ipiv;
   int                       curr, i, iroff, jb, lda, ldl2, mp, n, nb,
                             nq0, nn, test;
   static int                tswap = 0;
   static HPL_T_SWAP         fswap = HPL_NO_SWP;
#define LDU                  n
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_UPDATE );
#endif
   nb = PANEL->nb; jb = PANEL->jb; n = PANEL->nq; lda = PANEL->lda;
   if( NN >= 0 ) n = Mmin( NN, n );
/*
 * There is nothing to update, enforce the panel broadcast.
 */
   if( ( n <= 0 ) || ( jb <= 0 ) )
   {
      if( PBCST != NULL )
      {
         do { (void) HPL_bcast( PBCST, IFLAG ); }
         while( *IFLAG != HPL_SUCCESS );
      }
#ifdef HPL_DETAILED_TIMING
      HPL_ptimer( HPL_TIMING_UPDATE );
#endif
      return;
   }
/*
 * Enable/disable the column panel probing mechanism
 */
   (void) HPL_bcast( PBCST, &test );
/*
 * 1 x Q case
 */
   if( PANEL->grid->nprow == 1 )
   {
      Aptr = PANEL->A;       L2ptr = PANEL->L2;   L1ptr = PANEL->L1;
      ldl2 = PANEL->ldl2;    dpiv  = PANEL->DPIV; ipiv  = PANEL->IWORK;
      mp   = PANEL->mp - jb; iroff = PANEL->ii;   nq0   = 0; 
      for( i = 0; i < jb; i++ ) { ipiv[i] = (int)(dpiv[i]) - iroff; }
/*
 * So far we have not updated anything -  test availability of the panel
 * to be forwarded - If detected forward it and finish the update in one
 * step.
 */
      while ( test == HPL_KEEP_TESTING )
      {
         nn = n - nq0; nn = Mmin( nb, nn );
/*
 * Update nb columns at a time
 */
#ifdef HPL_DETAILED_TIMING
         HPL_ptimer( HPL_TIMING_LASWP );
         HPL_dlaswp00N( jb, nn, Aptr, lda, ipiv );
         HPL_ptimer( HPL_TIMING_LASWP );
#else
         HPL_dlaswp00N( jb, nn, Aptr, lda, ipiv );
#endif
#ifdef HPL_CUDA_DIAGNOSTICS
         HPL_fprintf(stderr, "HPL_dpupdateTT.c {1xQ while}: dtrsmLUTU (m=%d,n=%d), dgemmNN (m=%d,n=%d,k=%d)\n", jb, nn, mp, nn, jb);
#endif
         HPL_dtrsm( HplColumnMajor, HplLeft, HplUpper, HplTrans,
                    HplUnit, jb, nn, HPL_rone, L1ptr, jb, Aptr, lda );
         HPL_dgemm( HplColumnMajor, HplNoTrans, HplNoTrans, mp, nn,
                    jb, -HPL_rone, L2ptr, ldl2, Aptr, lda, HPL_rone,
                    Mptr( Aptr, jb, 0, lda ), lda );

         Aptr = Mptr( Aptr, 0, nn, lda ); nq0 += nn; 

         (void) HPL_bcast( PBCST, &test ); 
      }
/*
 * The panel has been forwarded at that point, finish the update
 */
      if( ( nn = n - nq0 ) > 0 )
      {
#ifdef HPL_DETAILED_TIMING
         HPL_ptimer( HPL_TIMING_LASWP );
         HPL_dlaswp00N( jb, nn, Aptr, lda, ipiv );
         HPL_ptimer( HPL_TIMING_LASWP );
#else
         HPL_dlaswp00N( jb, nn, Aptr, lda, ipiv );
#endif
         struct gpuUpdatePlan * plan = gpuUpdatePlanCreate(mp, nn, jb);

         if (plan->strategy != gpuNone) {
            gpu_upload(jb, nn, &(plan->gA), Aptr, lda);
            gpu_upload(jb, jb, &(plan->gL1), L1ptr, jb);

            cublasDtrsm('L','U','T','U', jb, nn, HPL_rone, 
                    plan->gL1.ptr, plan->gL1.lda, plan->gA.ptr, plan->gA.lda );
            gpuQ( cublasGetError() );

            gpu_download(jb, nn, Aptr, lda, &(plan->gA));
            gpu_upload(mp, jb, &(plan->gL2), L2ptr, ldl2);

            int iternum=0;
            int N0 = 0;

            while (N0 < nn) {
                int N1, N2;
                gpuDgemmPlanStage(plan, iternum, &N1, &N2);

#ifdef HPL_CUDA_DIAGNOSTICS
                HPL_fprintf(stderr, "HPL_dpupdateTT.c {1xQ if}: dtrsmLUTU (m=%5d,n=%5d), dgemmNN (m=%5d,n=%5d,k=%5d,lda=%5d,ldb=%5d,ldc=%5d) [i=%d,n0=%5d,n1=%5d,n2=%5d]\n", 
                            jb, nn, mp, nn, jb, plan->gL2.lda, plan->gA.lda, plan->gA2.lda, iternum, N0, N1, N2);
#endif

                gpu_upload(mp, N1, &(plan->gA2), Mptr(Aptr, jb, N0, lda), lda);

                cublasDgemm('N','N', mp, N1, jb, 
                        -HPL_rone, plan->gL2.ptr, plan->gL2.lda, 
                         Mptr( plan->gA.ptr, 0, N0, plan->gA.lda ), plan->gA.lda, 
                         HPL_rone, plan->gA2.ptr, plan->gA2.lda );
                gpuQ( cublasGetError() );

                HPL_dgemm( HplColumnMajor, HplNoTrans, HplNoTrans, mp, N2, jb,
                        -HPL_rone, L2ptr, ldl2, Mptr( Aptr, 0, N0+N1, lda ), lda,
                         HPL_rone, Mptr( Aptr, jb, N0+N1, lda ), lda );

                gpu_download(mp, N1, Mptr( Aptr, jb, N0, lda), lda, &(plan->gA2));

                N0 += N1 + N2;
                iternum++;
            }

         } else {
             HPL_dtrsm( HplColumnMajor, HplLeft, HplUpper, HplTrans,
                        HplUnit, jb, nn, HPL_rone, L1ptr, jb, Aptr, lda );
             HPL_dgemm( HplColumnMajor, HplNoTrans, HplNoTrans, mp, nn,
                        jb, -HPL_rone, L2ptr, ldl2, Aptr, lda, HPL_rone,
                        Mptr( Aptr, jb, 0, lda ), lda );
         }
         gpuUpdatePlanDestroy(plan);
      }
   }
   else                        /* nprow > 1 ... */
   {
/*
 * Selection of the swapping algorithm - swap:broadcast U.
 */
      if( fswap == HPL_NO_SWP )
      { fswap = PANEL->algo->fswap; tswap = PANEL->algo->fsthr; }

      if( (   fswap == HPL_SWAP01 ) ||
          ( ( fswap == HPL_SW_MIX ) && ( n > tswap ) ) )
      { HPL_pdlaswp01T( PBCST, &test, PANEL, n ); }
      else
      { HPL_pdlaswp00T( PBCST, &test, PANEL, n ); }
/*
 * Compute redundantly row block of U and update trailing submatrix
 */
      nq0 = 0; curr = ( PANEL->grid->myrow == PANEL->prow ? 1 : 0 );
      Aptr = PANEL->A; L2ptr = PANEL->L2;  L1ptr = PANEL->L1;
      Uptr = PANEL->U; ldl2 = PANEL->ldl2;
      mp   = PANEL->mp - ( curr != 0 ? jb : 0 );
/*
 * Broadcast has not occured yet, spliting the computational part
 */
      while ( test == HPL_KEEP_TESTING )
      {
         nn = n - nq0; nn = Mmin( nb, nn );

#ifdef HPL_CUDA_DIAGNOSTICS
         HPL_fprintf(stderr, "HPL_dpupdateTT.c {PxQ while}: dtrsmLUTU (m=%d,n=%d), dgemmNN (m=%d,n=%d,k=%d)\n", jb, nn, mp, nn, jb);
#endif
         HPL_dtrsm( HplColumnMajor, HplRight, HplUpper, HplNoTrans,
                    HplUnit, nn, jb, HPL_rone, L1ptr, jb, Uptr, LDU );

         if( curr != 0 )
         {
            HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                       Mptr( Aptr, jb, 0, lda ), lda );
            HPL_dlatcpy( jb, nn, Uptr, LDU, Aptr, lda );
         }
         else
         {
            HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                       Aptr, lda );
         }
         Uptr = Mptr( Uptr, nn, 0, LDU );
         Aptr = Mptr( Aptr, 0, nn, lda ); nq0 += nn;

         (void) HPL_bcast( PBCST, &test ); 
      }
/*
 * The panel has been forwarded at that point, finish the update
 */
      if( ( nn = n - nq0 ) > 0 )
      {
#ifdef HPL_CUDA_DIAGNOSTICS
        HPL_fprintf(stderr, "HPL_dpupdateTT.c {PxQ if}: dtrsmLUTU (m=%d,n=%d), dgemmNN (m=%d,n=%d,k=%d)\n", 
                    jb, nn, mp, nn, jb);
#endif
         HPL_dtrsm( HplColumnMajor, HplRight, HplUpper, HplNoTrans,
                    HplUnit, nn, jb, HPL_rone, L1ptr, jb, Uptr, LDU );

         if( curr != 0 )
         {
            HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                       Mptr( Aptr, jb, 0, lda ), lda );
            HPL_dlatcpy( jb, nn, Uptr, LDU, Aptr, lda );
         }
         else
         {
            HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                       Aptr, lda );
         }
      }
   }

   PANEL->A = Mptr( PANEL->A, 0, n, lda ); PANEL->nq -= n; PANEL->jj += n;
/*
 * return the outcome of the probe  (should always be  HPL_SUCCESS,  the
 * panel broadcast is enforced in that routine).
 */
   if( PBCST != NULL ) *IFLAG = test;
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer( HPL_TIMING_UPDATE );
#endif
/*
 * End of HPL_pdupdateTT
 */
}
/* vim:ts=4:sw=4:expandtab:number
 */
