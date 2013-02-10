#include "dist_tensor.hpp"
//#include "config.h"
#include <algorithm>
#include <iomanip>
#include <ostream>
#include <iostream>

using namespace std;

extern "C" {
#include "tensor.h"
#include "../libtensor/reference/util.h"
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
}

#include "cyclopstf.hpp"
#include "spinorbital.hpp"

#ifdef ENABLE_ASSERT
#define DTASSERT(...) assert(__VA_ARGS__)
#else
#define DTASSERT(...)
#endif

template<typename T>
int conv_idx(int const  ndim,
             T const *  cidx,
             int **     iidx){
  int i, j, n;
  T c;

  *iidx = (int*)malloc(sizeof(int)*ndim);

  n = 0;
  for (i=0; i<ndim; i++){
    c = cidx[i];
    for (j=0; j<i; j++){
      if (c == cidx[j]){
        (*iidx)[i] = (*iidx)[j];
        break;
      }
    }
    if (j==i){
      (*iidx)[i] = n;
      n++;
    }
  }
  return n;
}

template<typename T>
int  conv_idx(int const         ndim_A,
              T const *         cidx_A,
              int **            iidx_A,
              int const         ndim_B,
              T const *         cidx_B,
              int **            iidx_B){
  int i, j, n;
  T c;

  *iidx_B = (int*)malloc(sizeof(int)*ndim_B);

  n = conv_idx(ndim_A, cidx_A, iidx_A);
  for (i=0; i<ndim_B; i++){
    c = cidx_B[i];
    for (j=0; j<ndim_A; j++){
      if (c == cidx_A[j]){
        (*iidx_B)[i] = (*iidx_A)[j];
        break;
      }
    }
    if (j==ndim_A){
      for (j=0; j<i; j++){
        if (c == cidx_B[j]){
          (*iidx_B)[i] = (*iidx_B)[j];
          break;
        }
      }
      if (j==i){
        (*iidx_B)[i] = n;
        n++;
      }
    }
  }
  return n;
}


template<typename T>
int  conv_idx(int const         ndim_A,
              T const *         cidx_A,
              int **            iidx_A,
              int const         ndim_B,
              T const *         cidx_B,
              int **            iidx_B,
              int const         ndim_C,
              T const *         cidx_C,
              int **            iidx_C){
  int i, j, n;
  T c;

  *iidx_C = (int*)malloc(sizeof(int)*ndim_C);

  n = conv_idx(ndim_A, cidx_A, iidx_A,
               ndim_B, cidx_B, iidx_B);

  for (i=0; i<ndim_C; i++){
    c = cidx_C[i];
    for (j=0; j<ndim_B; j++){
      if (c == cidx_B[j]){
        (*iidx_C)[i] = (*iidx_B)[j];
        break;
      }
    }
    if (j==ndim_B){
      for (j=0; j<ndim_A; j++){
        if (c == cidx_A[j]){
          (*iidx_C)[i] = (*iidx_A)[j];
          break;
        }
      }
      if (j==ndim_A){
        for (j=0; j<i; j++){
          if (c == cidx_C[j]){
            (*iidx_C)[i] = (*iidx_C)[j];
            break;
          }
        }
        if (j==i){
          (*iidx_C)[i] = n;
          n++;
        }
      }
    }
  }
  return n;
}


DistWorld::DistWorld(int const inner_sz, MPI_Comm comm_){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new CTF();
#ifdef BGQ
  ctf->init(comm, MACHINE_BGQ, rank, np, inner_sz);
#else
#ifdef BGP
  ctf->init(comm, MACHINE_BGP, rank, np, inner_sz);
#else
  ctf->init(comm, MACHINE_8D, rank, np, inner_sz);
#endif
#endif
}

DistWorld::DistWorld(int const ndim, int const * lens, MPI_Comm comm_){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new CTF();
  ctf->init(comm, rank, np, ndim, lens);
}

DistWorld::~DistWorld(){
  delete ctf;
}

namespace libtensor
{
    template<>
    double scalar(const IndexedTensor<DistTensor>& other)
    {
        DistTensor dt(0, NULL, NULL, other.tensor_.dw);
        int n;
        double ret, * val;
        dt[""] = other;
        dt.getAllData(&n, &val);
        assert(n==1);
        ret = val[0];
        free(val);
        return ret;
    }

    template<>
    double scalar(const IndexedTensorMult<DistTensor>& other)
    {
        DistTensor dt(0, NULL, NULL, other.A_.tensor_.dw);
        int n;
        double ret, * val;
        dt[""] = other;
        dt.getAllData(&n, &val);
        assert(n==1);
        ret = val[0];
        free(val);
        return ret;
    }

    template<>
    double scalar(const IndexedTensor< SpinorbitalTensor<DistTensor> >& other)
    {
        DistTensor dt(0, NULL, NULL, other.tensor_.getSpinCase(0).dw);
        SpinorbitalTensor<DistTensor> sodt(",");
        sodt.addSpinCase(dt, ",", "");
        int n;
        double ret, * val;
        sodt[""] = other;
        dt.getAllData(&n, &val);
        assert(n==1);
        ret = val[0];
        free(val);
        return ret;
    }

    template<>
    double scalar(const IndexedTensorMult< SpinorbitalTensor<DistTensor> >& other)
    {
        DistTensor dt(0, NULL, NULL, other.A_.tensor_.getSpinCase(0).dw);
        SpinorbitalTensor<DistTensor> sodt(",");
        sodt.addSpinCase(dt, ",", "");
        int n;
        double ret, * val;
        sodt[""] = other;
        dt.getAllData(&n, &val);
        assert(n==1);
        ret = val[0];
        free(val);
        return ret;
    }
}

DistTensor::DistTensor(const DistTensor &       A) : Tensor<DistTensor>(A.ndim_, A.len_){
    int ndim, ret;
    int * edge_len;

    dw = A.dw;

    ret = dw->ctf->info_tensor(A.tid, &ndim, &edge_len, &sym);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);

    ret = dw->ctf->define_tensor(ndim, edge_len, sym, &tid);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);

    ret = dw->ctf->copy_tensor(A.tid, tid);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
    free(edge_len);
}


DistTensor::DistTensor(const DistTensor&        A,
                       DistWorld *              _dw,
                       const bool               copy,
                       const bool               zero) : Tensor<DistTensor>(A.ndim_, A.len_) {
    int ndim, ret;
    int * edge_len;

    dw = _dw;

    ret = dw->ctf->info_tensor(A.tid, &ndim, &edge_len, &sym);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);

    ret = dw->ctf->define_tensor(ndim, edge_len, sym, &tid);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);

    if (copy)
    {
        ret = dw->ctf->copy_tensor(A.tid, tid);
        DTASSERT(ret == DIST_TENSOR_SUCCESS);
    }
    free(edge_len);
}

DistTensor::DistTensor(const int        ndim,
                       const int *      len,
                       const int *      sym_,
                       DistWorld *      _dw,
                       const bool       zero) : Tensor<DistTensor>(ndim, len) {
    int ret;
    dw = _dw;

    sym = (int*)malloc(ndim*sizeof(int));

    memcpy(sym, sym_, sizeof(int)*ndim);

#ifdef VALIDATE_INPUTS
    validate_tensor(ndim, len, NULL, sym);
#endif //VALIDATE_INPUTS

    ret = dw->ctf->define_tensor(ndim, len, sym, &tid);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

DistTensor::~DistTensor(){
//    printf("Deleting tensor %d\n",tid);
    free(sym);
    dw->ctf->clean_tensor(tid);
}

/*
 * These are not needed (unless dw->ctf->get_dimension() gives something besides ndim_?)
 *
int DistTensor::getDimension() const
{
    int ndim;
    dw->ctf->get_dimension(tid,&ndim);
    return ndim;
}

const int* DistTensor::getLengths() const
{
    int *edge_len;
    dw->ctf->get_lengths(tid,&edge_len);
    return edge_len;
}

const int* DistTensor::getLeadingDims() const
{
    int *edge_len;
    dw->ctf->get_lengths(tid,&edge_len);
    return edge_len;
}
*/

double* DistTensor::getRawData(int * size)
{
    int ret;
    double * data;
    ret = dw->ctf->get_raw_data(tid, &data, size);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
    return data;
}

const double* DistTensor::getRawData(int * size) const
{
    int ret;
    double * data;
    ret = dw->ctf->get_raw_data(tid, &data, size);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
    return data;
}

void DistTensor::getLocalData(int * npair, kv_pair ** pairs) const
{
    int ret;
    ret = dw->ctf->read_local_tensor(tid, npair, pairs);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::getRemoteData(int npair, kv_pair * pairs) const
{
    int ret;
    ret = dw->ctf->read_tensor(tid, npair, pairs);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::writeRemoteData(int npair, kv_pair * pairs)
{
    int ret;
    ret = dw->ctf->write_tensor(tid, npair, pairs);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::getAllData(int * npair, double ** vals) const
{
    int ret;
    uint64_t unpair;
    ret = dw->ctf->allread_tensor(tid, &unpair, vals);
    *npair = (int)unpair;
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}


const int* DistTensor::getSymmetry() const
{
    return sym;
}



void DistTensor::mult(const double      alpha,
                      const DistTensor& A,
                      const int*        idx_A,
                      const DistTensor& B,
                      const int*        idx_B,
                      const double      beta,
                      const int*        idx_C)
{
    int ret;
    CTF_ctr_type_t tp;
    tp.tid_A = A.tid;
    tp.tid_B = B.tid;
    tp.tid_C = tid;
    conv_idx(A.ndim_, idx_A, &tp.idx_map_A,
             B.ndim_, idx_B, &tp.idx_map_B,
             ndim_, idx_C, &tp.idx_map_C);
   // dw->ctf->print_ctr(&tp);
   /* fseq_tsr_ctr<double> fs;
    fs.func_ptr = tensor_mult_;
    ret = dw->ctf->contract(&tp, NULL, NULL, fs, alpha, beta);*/
    ret = dw->ctf->contract(&tp, alpha, beta);

    free(tp.idx_map_A);
    free(tp.idx_map_B);
    free(tp.idx_map_C);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::contract(const double          alpha,
                          const DistTensor&     A,
                          const int*            idx_A,
                          const DistTensor&     B,
                          const int*            idx_B,
                          const double          beta,
                          const int*            idx_C)
{
    int ret;
    CTF_ctr_type_t tp;
    tp.tid_A = A.tid;
    tp.tid_B = B.tid;
    tp.tid_C = tid;
    conv_idx(A.ndim_, idx_A, &tp.idx_map_A,
             B.ndim_, idx_B, &tp.idx_map_B,
             ndim_, idx_C, &tp.idx_map_C);
    ret = dw->ctf->contract(&tp, alpha, beta);
    free(tp.idx_map_A);
    free(tp.idx_map_B);
    free(tp.idx_map_C);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::weight(const double            alpha,
                        const DistTensor&       A,
                        const int*              idx_A,
                        const DistTensor&       B,
                        const int*              idx_B,
                        const double            beta,
                        const int*              idx_C)
{
    int ret;
    CTF_ctr_type_t tp;
    tp.tid_A = A.tid;
    tp.tid_B = B.tid;
    tp.tid_C = tid;
    conv_idx(A.ndim_, idx_A, &tp.idx_map_A,
             B.ndim_, idx_B, &tp.idx_map_B,
             ndim_, idx_C, &tp.idx_map_C);
    fseq_tsr_ctr<double> fs;
    fs.func_ptr = tensor_mult_;
    ret = dw->ctf->contract(&tp, NULL, 0, fs, alpha, beta);
    free(tp.idx_map_A);
    free(tp.idx_map_B);
    free(tp.idx_map_C);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::outerProduct(const double              alpha,
                              const DistTensor&         A,
                              const int*                idx_A,
                              const DistTensor&         B,
                              const int*                idx_B,
                              const double              beta,
                              const int*                idx_C)
{
    int ret;
    CTF_ctr_type_t tp;
    tp.tid_A = A.tid;
    tp.tid_B = B.tid;
    tp.tid_C = tid;
    conv_idx(A.ndim_, idx_A, &tp.idx_map_A,
             B.ndim_, idx_B, &tp.idx_map_B,
             ndim_, idx_C, &tp.idx_map_C);
    fseq_tsr_ctr<double> fs;
    fs.func_ptr = tensor_outer_prod_;
    ret = dw->ctf->contract(&tp, NULL, 0, fs, alpha, beta);
    free(tp.idx_map_A);
    free(tp.idx_map_B);
    free(tp.idx_map_C);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}


DistTensor& DistTensor::operator=(const double val){
    int size;
    double* raw_data = getRawData(&size);
    fill(raw_data, raw_data+size, val);
    return *this;
}

void DistTensor::print(FILE* fp) const
{
    dw->ctf->print_tensor(fp, tid);
}

void DistTensor::transpose(const double         alpha,
                           const DistTensor&    A,
                           const int*           idx_A,
                           const double         beta,
                           const int*           idx_B)
{
    int ret;
    int * idx_map_A, * idx_map_B;
    conv_idx(A.ndim_, idx_A, &idx_map_A,
             ndim_, idx_B, &idx_map_B);
    fseq_tsr_sum<double> fs;
    fs.func_ptr = tensor_transpose_;
    ret = dw->ctf->sum_tensors(alpha, beta, A.tid, tid, idx_map_A, idx_map_B, fs);
    free(idx_map_A);
    free(idx_map_B);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::sum(const double       alpha,
                     const DistTensor&  A,
                     const int*         idx_A,
                     const double       beta,
                     const int*         idx_B)
{
    int ret;
    int * idx_map_A, * idx_map_B;
    CTF_sum_type_t st;
    conv_idx(A.ndim_, idx_A, &idx_map_A,
             ndim_, idx_B, &idx_map_B);
/*    fseq_tsr_sum<double> fs;
    fs.func_ptr = tensor_sum_;*/
    st.idx_map_A = idx_map_A;
    st.idx_map_B = idx_map_B;
    st.tid_A = A.tid;
    st.tid_B = tid;
//    dw->ctf->print_sum(&st);
    ret = dw->ctf->sum_tensors(&st, alpha, beta);
    free(idx_map_A);
    free(idx_map_B);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::diagonal(const double          alpha,
                          const DistTensor&     A,
                          const int*            idx_A,
                          const double          beta,
                          const int*            idx_B)
{
    int ret;
    int * idx_map_A, * idx_map_B;
    CTF_sum_type_t st;
    conv_idx(A.ndim_, idx_A, &idx_map_A,
             ndim_, idx_B, &idx_map_B);
    st.idx_map_A = idx_map_A;
    st.idx_map_B = idx_map_B;
    st.tid_A = A.tid;
    st.tid_B = tid;
    fseq_tsr_sum<double> fs;
    fs.func_ptr = tensor_diagonal_;
    ret = dw->ctf->sum_tensors(&st, alpha, beta, fs);
    free(idx_map_A);
    free(idx_map_B);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::replicate(const double         alpha,
                           const DistTensor&    A,
                           const int*           idx_A,
                           const double         beta,
                           const int*           idx_B)
{
    int ret;
    int * idx_map_A, * idx_map_B;
    conv_idx(A.ndim_, idx_A, &idx_map_A,
             ndim_, idx_B, &idx_map_B);
    fseq_tsr_sum<double> fs;
    fs.func_ptr = tensor_replicate_;
    ret = dw->ctf->sum_tensors(alpha, beta, A.tid, tid, idx_map_A, idx_map_B, fs);
    free(idx_map_A);
    free(idx_map_B);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::trace(const double alpha, const DistTensor& A, const int* idx_A,
                       const double beta,                    const int* idx_B)
{
    int ret;
    int * idx_map_A, * idx_map_B;
    conv_idx(A.ndim_, idx_A, &idx_map_A,
         ndim_, idx_B, &idx_map_B);
    fseq_tsr_sum<double> fs;
    fs.func_ptr = tensor_trace_;
    ret = dw->ctf->sum_tensors(alpha, beta, A.tid, tid, idx_map_A, idx_map_B, fs);
    free(idx_map_A);
    free(idx_map_B);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

void DistTensor::resym(const double alpha, const DistTensor& A, const int* idx_A,
                       const double beta,                    const int* idx_B)
{
    DTASSERT(0);
}

double DistTensor::reduce(CTF_OP op){
  int ret;
  double ans;
  ans = 0.0;
  ret = dw->ctf->reduce_tensor(tid, op, &ans);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
  return ans;
}

void DistTensor::pack(const DistTensor& A)
{
    //not needed for now
}

void DistTensor::symmetrize(const DistTensor& A)
{
    //not needed for now
}

void DistTensor::scale(const double alpha, const int* idx_A)
{
  int ret;
  int * idx_map_A;
//    assert(0);
  conv_idx(ndim_, idx_A, &idx_map_A);
//    fseq_tsr_scl<double> fs;
//    fs.func_ptr = tensor_scale_;
  ret = dw->ctf->scale_tensor(alpha, tid, idx_map_A);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);

}

const DistTensor& DistTensor::slice(const int* start, const int* len) const
{
    //TODO
    return *this;
}

/*
 * This would be really nice to have, but perhaps not critical
 *
 * Let me know if you would like me to explain what it should do
 */
DistTensor& DistTensor::slice(const int* start, const int* len)
{
    //TODO
    return *this;
}
