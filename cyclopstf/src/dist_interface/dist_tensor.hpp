#ifndef _DIST_TENSOR_HPP_
#define _DIST_TENSOR_HPP_

#include <ostream>

//extern "C" {
#include <stdio.h>
#include <stdint.h>
//}

#include "cyclopstf.hpp"
#include "tensor.hpp"

using namespace libtensor;

class DistWorld
{
    public:
        CTF * ctf;
        MPI_Comm comm;

    public:
        DistWorld(int const inner_sz = DEF_INNER_SIZE, MPI_Comm comm_ = MPI_COMM_WORLD);
        DistWorld(int const ndim, int const * lens, MPI_Comm comm_ = MPI_COMM_WORLD);
        ~DistWorld();

};


class DistTensor;
template <class Base> class SpinorbitalTensor;

namespace libtensor
{
    template<>
    double scalar(const IndexedTensor<DistTensor>& other);

    template<>
    double scalar(const IndexedTensorMult<DistTensor>& other);

    template<>
    double scalar(const IndexedTensor< SpinorbitalTensor<DistTensor> >& other);

    template<>
    double scalar(const IndexedTensorMult< SpinorbitalTensor<DistTensor> >& other);
}

//template<class DistTensor>
//double scalar(const IndexedTensor<DistTensor>& other);
//template<class DistTensor>
//double scalar(const IndexedTensorMult<DistTensor>& other);

class DistTensor : public Tensor<DistTensor>
{
    friend class PackedTensor;
    friend class DenseTensor;

//    friend double scalar(const IndexedTensor<DistTensor>& other);
//    friend double scalar(const IndexedTensorMult<DistTensor>& other);

//    friend double scalar< SpinorbitalTensor<DistTensor> >(const IndexedTensor< SpinorbitalTensor<DistTensor> >& other);
//    friend double scalar< SpinorbitalTensor<DistTensor> >(const IndexedTensorMult< SpinorbitalTensor<DistTensor> >& other);
//    friend double scalar(const IndexedTensor< SpinorbitalTensor<DistTensor> >& other);
//    friend double scalar(const IndexedTensorMult< SpinorbitalTensor<DistTensor> >& other);

    INHERIT_FROM_TENSOR(DistTensor)

    /*
     * Not needed anymore, just don't implement
     *
     * private:
         * double* getData() = 0;
     */

    public:
        int tid;
        DistWorld * dw;
        int* sym;
        size_t localSize;

    public:
            /*
             * You should probably also include a traditional copy ctor so that \
             * the compiler doesn't generate a stupid one for you.
             * (or you could make it private)
             *
             * i.e. DistTensor(const DistTensor& A)
             */
        DistTensor(const DistTensor &   A);

        DistTensor(const DistTensor &   A,
                   DistWorld *          _dw,
                   const bool           copy = false,
                   const bool           zero = true);

        DistTensor(const int            ndim,
                   const int *          len,
                   const int *          sym,
                   DistWorld *          _dw,
                   const bool           zero=true);

        ~DistTensor();


        double* getRawData(int * size);

    const double* getRawData(int * size) const;

  void getLocalData(int * npair, kv_pair ** pairs) const;

	void getRemoteData(int npair, kv_pair * pairs) const;

  void getAllData(int * npair, double ** vals) const;

        void writeRemoteData(int npair, kv_pair * pairs);

        const int* getSymmetry() const;

        const int* getSymmetryType() const;

        void mult(const double alpha, const DistTensor& A, const int* idx_A,
                                      const DistTensor& B, const int* idx_B,
                  const double beta,                       const int* idx_C);

        void contract(const double alpha, const DistTensor& A, const int* idx_A,
                                          const DistTensor& B, const int* idx_B,
                      const double beta,                       const int* idx_C);

        void weight(const double alpha, const DistTensor& A, const int* idx_A,
                                        const DistTensor& B, const int* idx_B,
                    const double beta,                       const int* idx_C);

        void outerProduct(const double alpha, const DistTensor& A, const int* idx_A,
                                              const DistTensor& B, const int* idx_B,
                          const double beta,                       const int* idx_C);

        virtual DistTensor& operator=(const double val);
        /*
         * These two are not necessary
         */
        void print(FILE* fp) const;
        void print(std::ostream& stream) const;

        void transpose(const double alpha, const DistTensor& A, const int* idx_A,
                       const double beta,                       const int* idx_B);

        void diagonal(const double alpha, const DistTensor& A, const int* idx_A,
                      const double beta,                       const int* idx_B);

        void replicate(const double alpha, const DistTensor& A, const int* idx_A,
                       const double beta,                    const int* idx_B);


        void sum(const double alpha, const DistTensor& A, const int* idx_A,
                 const double beta,                       const int* idx_B);

        virtual void resym(const double alpha, const DistTensor& A, const char* idx_A,
                           const double beta,                       const char* idx_B)
        {
            int *idx_A_ = new int[A.ndim_];
            int *idx_B_ = new int[ndim_];

            for (int i = 0;i < A.ndim_;i++) idx_A_[i] = idx_A[i];
            for (int i = 0;i < ndim_;i++) idx_B_[i] = idx_B[i];

            resym(alpha, A, idx_A_,
                   beta,    idx_B_);

            delete[] idx_A_;
            delete[] idx_B_;
        }

        virtual void resym(const double alpha, const DistTensor& A, const int* idx_A,
                           const double beta,                       const int* idx_B);

        void trace(const double alpha, const DistTensor& A, const int* idx_A,
                   const double beta,                       const int* idx_B);

        double reduce(CTF_OP op);

        /*
         * These two are not necessary unless you really want to implement them
         */
        void pack(const DistTensor& A);
        void symmetrize(const DistTensor& A);

        //using Tensor<DistTensor>::scale;

        void scale(const double alpha, const int* idx_A);

        /*
         * These two would also be an "experimental" feature
         */
        DistTensor& slice(const int* start, const int* len);
        const DistTensor& slice(const int* start, const int* len) const;
};

#endif
