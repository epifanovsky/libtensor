/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#include "../dist_tensor/dist_tensor.h"
#include "../dist_tensor/dist_tensor_internal.h"
#include "../shared/util.h"
#include "../shared/unit_util.h"
#include "unit_test.h"
#include "unit_test_ctr.h"
#include "../shared/test_symmetry.hxx"
#include "test_sym_contract.h"

#ifndef TSR_EDGE_LEN
#define TSR_EDGE_LEN    7
#endif


int get_perm_index(int const ndim, int const * arr){
  int i,j,t;
  t = 0;
  for (i=0; i<ndim-1; i++){
    t = t * (ndim-i);
    for (j=i+1; j<ndim; j++){
      if (arr[i] > arr[j]) t++;
    }
  }
  return t;
}

void get_perm(int const ndim, int const t1, int * arr){
  int i,j,t;
  t=t1;
  arr[ndim-1]=1;
  for (i=ndim-1; i>1; i--){
    arr[i] = 1 + (t % (ndim-i));
    t = t / (ndim-i);
    for (j=i+1; j<ndim; j++){
      if (arr[j] >= arr[i]) arr[j] = arr[j] + 1;
    }
  }
}

void permute(int const ndim, int const * perm, int * arr){
  int i;
  int * tmp;
  
  tmp = (int*)malloc(ndim*sizeof(int));
  memcpy(tmp,arr,ndim*sizeof(int));

  for (i=0; i<ndim; i++){
    arr[perm[i]] = tmp[i];
  }
  free(tmp);
}

int factorial(int const n){
  int i,a;
  a=1;
  for (i=1; i<=n; i++){
    a*=i;
  }
  return a;
}

int inc_map(int const ndim,
            int const ntot,
            int * map){
  int i,min;
  int * all_map;

  all_map = (int*)calloc(sizeof(int)*ntot,1);

  for (i=0; i<ndim; i++){
    all_map[map[i]] = i+1;
  }  
}


int get_map(int const   ndim, 
            int const   nctr, 
            int const   ntot, 
            int *       idx_map,
            int const   init){
  int p, i;
  int * perm;

  if (init){
    get_perm(ndim, 0, idx_map);
    for (i=0; i<ndim; i++){
      printf("init idx_map[%d] = %d\n",i,idx_map[i]);
    }
    return 1;
  } else {
    /* FIXME: REDUCE INDEX MAP */
    p = get_perm_index(ndim, idx_map);
    p++;
    if (p==factorial(ndim)) p=0;
    perm = (int*)malloc(sizeof(int)*ndim);
    get_perm(ndim, p, perm);
    permute(ndim, perm, idx_map);
    free(perm);
    return 1;
  }
  if (p==0){
    return inc_map(ndim, ntot, idx_map);
  } else return 1;
}

void get_tensor(int const ndim, int * tid){
  int i;
  int * edge_len, * sym, * sym_type;

  edge_len      = (int*)malloc(sizeof(int)*ndim);
  sym           = (int*)malloc(sizeof(int)*ndim);
  sym_type      = (int*)malloc(sizeof(int)*ndim);

  std::fill(edge_len, edge_len+ndim, TSR_EDGE_LEN);
  std::fill(sym, sym+ndim, -1);
  std::fill(sym_type, sym_type+ndim, -1);

  CTF_define_tensor(ndim, edge_len, sym, sym_type, tid);
  
}

/**
 * \brief verifies correctness of symmetric operations
 */
void auto_contract(int const            argc, 
                    char **             argv, 
                    int const           numPes, 
                    int const           myRank, 
                    CommData_t *        cdt_glb){

  int seed, in_num, tid_A, tid_B, tid_C, pass, i, j, k, dim;
  int ntot, nctr, ndim_A, ndim_B, ndim_C, delta;
  char ** input_str;
  CTF_ctr_type_t ctype;

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }

  /*if (getCmdOption(input_str, input_str+in_num, "-nctr")){
    nctr = atoi(getCmdOption(input_str, input_str+in_num, "-nctr"));
    if (nctr < 0) nctr = 1;
  } else nctr = 1;*/
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;

//  read_topology(input_str, in_num, myRank, numPes);

  CTF_init(MPI_COMM_WORLD, myRank, numPes);
  
  ndim_A = 4;
  ndim_B = 4;
  ndim_C = 4;
  get_tensor(ndim_A, &tid_A);
  get_tensor(ndim_B, &tid_B);
  get_tensor(ndim_C, &tid_C);

  for (i=0; i<NUM_RHF_NONSYM; i++){
    ctype.idx_map_A     = maps_A[i];
    ctype.idx_map_B     = maps_B[i];
    ctype.idx_map_C     = maps_C[i];
    ctype.tid_A         = tid_A;
    ctype.tid_B         = tid_B;
    ctype.tid_C         = tid_C;
    GLOBAL_BARRIER(cdt_glb);
    if (myRank == 0)
      print_ctr(&ctype);
    GLOBAL_BARRIER(cdt_glb);

    test_sym_readwrite(seed, tid_A, myRank, numPes);
    test_sym_readwrite(seed, tid_B, myRank, numPes);
    test_sym_readwrite(seed, tid_C, myRank, numPes);

    pass = test_sym_contract(&ctype, myRank, numPes);
    GLOBAL_BARRIER(cdt_glb);
    if (myRank==0) {
      if (pass){
        printf("RHF nonsymmetric contraction test %d successfull.\n",i);
      } else {
        printf("RHF nonsymmetric contraction test %d FAILED!!!\n",i);
      }
    }

    j = tid_A;
    tid_A = tid_C;
    tid_B = tid_C;
    tid_C = j;
  }

  CTF_exit();

#if 0
  for (dim=4; dim<5; dim++){
    nctr = dim/2;
    ndim_A = dim;
    ndim_B = dim;
    ndim_C = dim;
    ntot = ndim_A+ndim_B-nctr;
    ctype.idx_map_A = (int*)calloc(sizeof(dim)*dim,1);
    ctype.idx_map_B = (int*)calloc(sizeof(dim)*dim,1);
    ctype.idx_map_C = (int*)calloc(sizeof(dim)*dim,1);
      
    
    i=0;
    while(true){
      delta = 0;
      delta = get_map(ndim_A, nctr, ntot, ctype.idx_map_A, i==0);
      if (delta == 0) break;
      j=0;
      while(true){
/*      delta = get_map(ndim_A, ndim_B, nctr, ntot, 
                        ctype.idx_map_A, ctype.idx_map_B, j==0);*/
        delta = get_map(ndim_B, nctr, ntot, ctype.idx_map_B, i==0);
        if (delta == 0) break;
/*      get_tensor(ndim_A, ndim_B, nctr, ntot, 
                   ctype.idx_map_A, ctype.idx_map_B, ctype.tid_A, ctype.tid_B);*/
        
        delta = get_map(ndim_C, nctr, ntot, ctype.idx_map_C, i==0);
        k=0;
        while(true){
/*        delta = get_map(ndim_A, ndim_B, nctr, ntot, 
                          ctype.idx_map_A, ctype.idx_map_B, k==0);*/
          if (delta == 0) break;
          /*get_tensor(ndim_A, ndim_B, ndim_C, nctr, ntot, 
                     ctype.idx_map_A, ctype.idx_map_B, ctype.idx_map_C, 
                     ctype.tid_A, ctype.tid_B, ctype.tid_C);*/

          print_ctr(&ctype);

          test_sym_readwrite(seed, tid_A, myRank, numPes);
          test_sym_readwrite(seed, tid_B, myRank, numPes);
          test_sym_readwrite(seed, tid_C, myRank, numPes);

          pass = test_sym_contract(&ctype, myRank, numPes);
          GLOBAL_BARRIER(cdt_glb);
          if (myRank==0) {
            if (pass){
              printf("Symmetric contraction test %d %d %d successfull.\n",i,j,k);
            } else {
              printf("Symmetric contraction test %d %d %d FAILED!!!\n",i,j,k);
            }
          }
          k++;
        }
        j++;
      }
      i++;
    }
  }
#endif

  GLOBAL_BARRIER(cdt_glb);
  if (myRank==0) {
    printf("Test auto completed\n");
  }
}

int main(int argc, char ** argv){
  int myRank, numPes;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);
  auto_contract(argc, argv, numPes, myRank, cdt_glb);
  COMM_EXIT;
  return 0;
}

