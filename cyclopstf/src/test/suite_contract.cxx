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
#include "test_sym_kernel.h"
#include "suite_rhf_nonsym.h"

#ifndef TSR_EDGE_LEN
#define TSR_EDGE_LEN    7
#endif


void get_tensor(int const ndim, int * tid){
  int i;
  int * edge_len, * sym;

  edge_len      = (int*)malloc(sizeof(int)*ndim);
  sym           = (int*)malloc(sizeof(int)*ndim);

  std::fill(edge_len, edge_len+ndim, TSR_EDGE_LEN);
  std::fill(sym, sym+ndim, 0);

  CTF_define_tensor(ndim, edge_len, sym, tid);
  
}

/**
 * \brief verifies correctness of symmetric operations
 */
void suite_contract(int const           argc, 
                    char **             argv, 
                    int const           numPes, 
                    int const           myRank, 
                    CommData_t *        cdt_glb){

  int seed, in_num, tid_A, tid_B, tid_C, pass, i, j, k, dim;
  int ntd, ntot, nctr, ndim_A, ndim_B, ndim_C, delta, norb, netron;
  int * tids;
  char ** input_str;
  CTF_ctr_type_t ctype;
  CTF_sum_type_t stype;
  RHF_step * st; 

  if (argc == 2) {
    read_param_file(argv[1], myRank, &input_str, &in_num);
  } else {
    input_str = argv;
    in_num = argc;
  }

  if (getCmdOption(input_str, input_str+in_num, "-norb")){
    norb = atoi(getCmdOption(input_str, input_str+in_num, "-norb"));
    if (norb < 0) norb = 12;
  } else norb = 12;
  if (getCmdOption(input_str, input_str+in_num, "-netron")){
    netron = atoi(getCmdOption(input_str, input_str+in_num, "-netron"));
    if (netron < 0) netron = 12;
  } else netron = 12;
  if (getCmdOption(input_str, input_str+in_num, "-seed")){
    seed = atoi(getCmdOption(input_str, input_str+in_num, "-seed"));
    if (seed < 0) seed = 3;
  } else seed = 3;

//  read_topology(input_str, in_num, myRank, numPes);

  CTF_init(MPI_COMM_WORLD, myRank, numPes);
  
/*  ndim_A = 4;
  ndim_B = 4;
  ndim_C = 4;
  get_tensor(ndim_A, &tid_A);
  get_tensor(ndim_B, &tid_B);
  get_tensor(ndim_C, &tid_C);*/
  
  create_tensors(norb, netron, &ntd, &tids);

  for (i=0; i<ntd; i++){    
    test_sym_readwrite(seed, tids[i], myRank, numPes);
  }

  for (i=0; i<NUM_RHF_NONSYM; i++){
    st = &sequence[i];

    switch (st->op){
      case ST_SCALE:
        printf("Not scaling anything ABORT\n");
        ABORT;
        break;
      
      case ST_SUM:
        stype.tid_A     = tids[st->tid_A];
        stype.tid_B     = tids[st->tid_B];
        stype.idx_map_A = st->map_A;
        stype.idx_map_B = st->map_B;
        GLOBAL_BARRIER(cdt_glb);
        if (myRank==0) {
          print_sum(&stype);
        }
        GLOBAL_BARRIER(cdt_glb);
        pass            = test_sym_sum(&stype, myRank, numPes, st->alpha, st->beta);
        GLOBAL_BARRIER(cdt_glb);
        if (myRank==0) {
          if (pass){
            printf("RHF nonsymmetric summation step %d successfull.\n",i);
          } else {
            printf("RHF nonsymmetric summation step %d FAILED!!!\n",i);
          }
        }
        GLOBAL_BARRIER(cdt_glb);
        break;
      
      case ST_CONTRACT:
        ctype.tid_A     = tids[st->tid_A];
        ctype.tid_B     = tids[st->tid_B];
        ctype.tid_C     = tids[st->tid_C];
        ctype.idx_map_A = st->map_A;
        ctype.idx_map_B = st->map_B;
        ctype.idx_map_C = st->map_C;
        GLOBAL_BARRIER(cdt_glb);
        if (myRank==0) {
          print_ctr(&ctype);
        }
        GLOBAL_BARRIER(cdt_glb);
        pass            = test_sym_contract(&ctype, myRank, numPes, st->alpha, st->beta);
        GLOBAL_BARRIER(cdt_glb);
        if (myRank==0) {
          if (pass){
            printf("RHF nonsymmetric contraction step %d successfull.\n",i);
          } else {
            printf("RHF nonsymmetric contraction step %d FAILED!!!\n",i);
          }
        }
        GLOBAL_BARRIER(cdt_glb);
        break;

      base:
        printf("the fuck is this?\n");
        ABORT;
        break;
    }

    GLOBAL_BARRIER(cdt_glb);


    /*j = tid_A;
    tid_A = tid_C;
    tid_B = tid_C;
    tid_C = j;*/
  }

  CTF_exit();

  GLOBAL_BARRIER(cdt_glb);
  if (myRank==0) {
    printf("RHF test suite completed\n");
  }
}

int main(int argc, char ** argv){
  int myRank, numPes;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);
  suite_contract(argc, argv, numPes, myRank, cdt_glb);
  COMM_EXIT;
  return 0;
}

