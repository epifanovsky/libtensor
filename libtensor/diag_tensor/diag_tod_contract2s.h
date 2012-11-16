#ifndef LIBTENSOR_DIAG_TOD_CONTRACT2S_H
#define LIBTENSOR_DIAG_TOD_CONTRACT2S_H

#include <list>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/dense_tensor/to_contract2_dims.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Contracts pairs of diagonal tensors (streaming)
    \param N Order of first tensor less contraction degree.
    \param M Order of second tensor less contraction degree.
    \param K Contraction degree.

    This is a streaming version of diag_tod_contract2 that contracts a series
    of pairs of arguments into one result.

    \sa diag_tod_contract2

    \ingroup libtensor_diag_tensor
 **/
template<size_t N, size_t M, size_t K>
class diag_tod_contract2s : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

public:
    struct args {
        contraction2<N, M, K> contr; //!< Contraction
        diag_tensor_rd_i<NA, double> &dta; //!< First tensor (A)
        diag_tensor_rd_i<NB, double> &dtb; //!< Second tensor (B)
        double d; //!< Scaling factor

        args(
            const contraction2<N, M, K> &contr_,
            diag_tensor_rd_i<NA, double> &dta_,
            diag_tensor_rd_i<NB, double> &dtb_,
            double d_) :
            contr(contr_), dta(dta_), dtb(dtb_), d(d_) { }
    };

private:
    to_contract2_dims<N, M, K> m_dimsc; //!< Dimensions of result
    std::list<args> m_argslst; //!< List of arguments

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param dta First contracted tensor A.
        \param ka Scalar transformation of A.
        \param dtb Second contracted tensor B.
        \param kb Scalar transformation of B.
        \param kc Scalar transformation of result (default 1.0).
     **/
    diag_tod_contract2s(
        const contraction2<N, M, K> &contr,
        diag_tensor_rd_i<NA, double> &dta,
        const scalar_transf<double> &ka,
        diag_tensor_rd_i<NB, double> &dtb,
        const scalar_transf<double> &kb,
        const scalar_transf<double> &kc = scalar_transf<double>());

    /** \brief Initializes the operation
        \param contr Contraction.
        \param dta First tensor (A).
        \param dtb Second tensor (B).
     **/
    diag_tod_contract2s(
        const contraction2<N, M, K> &contr,
        diag_tensor_rd_i<NA, double> &dta,
        diag_tensor_rd_i<NB, double> &dtb,
        double d = 1.0);

    /** \brief Adds a set of arguments to the argument list
        \param contr Contraction.
        \param dta First contracted tensor A.
        \param ka Scalar transformation of A.
        \param dtb Second contracted tensor B.
        \param kb Scalar transformation of B.
        \param kc Scalar transformation of result (C).
     **/
    void add_args(
        const contraction2<N, M, K> &contr,
        diag_tensor_rd_i<NA, double> &dta,
        const scalar_transf<double> &ka,
        diag_tensor_rd_i<NB, double> &dtb,
        const scalar_transf<double> &kb,
        const scalar_transf<double> &kc);

    /** \brief Adds a set of arguments to the argument list
        \param contr Contraction.
        \param dta First contracted tensor A.
        \param dtb Second contracted tensor B.
        \param d Scaling factor d.
     **/
    void add_args(
        const contraction2<N, M, K> &contr,
        diag_tensor_rd_i<NA, double> &dta,
        diag_tensor_rd_i<NB, double> &dtb,
        double d);

    /** \brief Performs the operation
        \param zero Zero output tensor before putting result.
        \param dtc Output tensor.
     **/
    void perform(bool zero, diag_tensor_wr_i<N + M, double> &dtc);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_CONTRACT2S_H

