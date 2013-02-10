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

#ifndef __SUITE_RHF_NONSYM__
#define __SUITE_RHF_NONSYM__


#define NUM_RHF_STEP    5
#define NUM_RHF_NONSYM  5

#include "../dist_tensor/dist_tensor.h"

void create_tensors(int const norb, int const netron, int * ntd, int ** tids){
  int i, ndim, tid;
  int * edge_len, * sym;

  *tids = (int*)malloc(sizeof(int)*5);
  *ntd = 5;

  /* F is 2D */
  ndim          = 2;
  edge_len      = (int*)malloc(sizeof(int)*ndim);
  sym           = (int*)malloc(sizeof(int)*ndim);
  
  edge_len[0] = netron;
  edge_len[1] = norb;
  std::fill(sym,        sym+ndim,       0);
  
  CTF_define_tensor(ndim, edge_len, sym, &tid);
  (*tids)[0] = tid;
  
  /* T1 is also 2D */
  CTF_define_tensor(ndim, edge_len, sym, &tid);
  (*tids)[1] = tid;
  
  /* V is 4D */
  ndim          = 4;
  edge_len      = (int*)malloc(sizeof(int)*ndim);
  sym           = (int*)malloc(sizeof(int)*ndim);
  
  edge_len[0] = netron;
  edge_len[1] = netron;
  edge_len[2] = norb;
  edge_len[3] = norb;
  std::fill(sym,        sym+ndim,       0);
  
  CTF_define_tensor(ndim, edge_len, sym, &tid);
  (*tids)[2] = tid;
  
  /* T2 is also 4D */
  CTF_define_tensor(ndim, edge_len, sym, &tid);
  (*tids)[3] = tid;

  /* W is also 4D */
  CTF_define_tensor(ndim, edge_len, sym, &tid);
  (*tids)[4] = tid;

}

enum step_op_type { ST_SCALE, ST_SUM, ST_CONTRACT };
typedef struct RHF_step_t {
  step_op_type op;

  int tid_A;
  int tid_B;
  int tid_C;

  int map_A[4];
  int map_B[4];
  int map_C[4];
  
  double alpha;
  double beta;
} RHF_step;

RHF_step sequence[NUM_RHF_STEP] = {
{ ST_SUM, 0, 0, 0,
  {0, 1, -1, -1}, {1, 0, -1, -1}, NULL,
  -1.0, 2.0},
{ ST_CONTRACT, 2, 1, 0,
  {0, 1, 2, 3}, {0, 2, -1, -1}, {1, 3, -1, -1},
  0.5, 1.0},
{ ST_SUM, 0, 0, 0,
  {0, 1, -1, -1}, {1, 0, -1, -1}, NULL,
  -1.0, 2.0},
{ ST_CONTRACT, 2, 3, 0,
  {0, 1, 2, 3}, {0, 4, 2, 3}, {1, 4, -1, -1},
  1.0, 1.0},
{ ST_SUM, 0, 0, 0,
  {0, 1, -1, -1}, {1, 0, -1, -1}, NULL,
  -1.0, 2.0}
};


int map_A1[4] = {0, 1, 2, 3};
int map_B1[4] = {0, 1, 4, 5};
int map_C1[4] = {2, 3, 4, 5};

int map_A2[4] = {0, 1, 2, 3};
int map_B2[4] = {0, 1, 4, 5};
int map_C2[4] = {3, 4, 5, 2};

int map_A3[4] = {0, 1, 2, 3};
int map_B3[4] = {4, 1, 2, 5};
int map_C3[4] = {3, 4, 5, 0};

int map_A4[4] = {0, 1, 2, 3};
int map_B4[4] = {4, 1, 2, 5};
int map_C4[4] = {5, 3, 0, 4};

int map_A5[4] = {0, 1, 2, 3};
int map_B5[4] = {4, 1, 2, 5};
int map_C5[4] = {5, 3, 0, 4};

int* maps_A[NUM_RHF_NONSYM] = {map_A1, map_A2, map_A3, map_A4, map_A5};
int* maps_B[NUM_RHF_NONSYM] = {map_B1, map_B2, map_B3, map_B4, map_B5};
int* maps_C[NUM_RHF_NONSYM] = {map_C1, map_C2, map_C3, map_C4, map_C5};


#endif// __SUITE_RHF_NONSYM__
