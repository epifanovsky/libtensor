#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_STREAMED_H
#define LIBTENSOR_CTF_TOD_CONTRACT2_STREAMED_H

#include <list>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Contracts two distributed tensors (streamed version)
    \tparam N Order of first tensor (A) less contraction degree.
    \tparam M Order of second tensor (B) less contraction degree.
    \tparam K Contraction degree (number of inner indexes).

    \sa tod_contract2, ctf_tod_contract2

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, size_t M, size_t K>
class ctf_tod_contract2_streamed : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

private:
    struct args {
        contraction2<N, M, K> contr; //!< Contraction
        ctf_dense_tensor_i<NA, double> &ta; //!< First tensor (A)
        ctf_dense_tensor_i<NB, double> &tb; //!< Second tensor (B)
        double d; //!< Scaling factor

        args(
            const contraction2<N, M, K> &contr_,
            ctf_dense_tensor_i<NA, double> &ta_,
            ctf_dense_tensor_i<NB, double> &tb_,
            double d_) :
            contr(contr_), ta(ta_), tb(tb_), d(d_) { }
    };

private:
    std::list<args> m_argslst; //!< List of arguments
    dimensions<NC> m_dimsc; //!< Dimensions of result

public:
    /** \brief Contracts two tensors
        \param contr Contraction.
        \param ta First contracted tensor A.
        \param ka Scalar transformation of A.
        \param tb Second contracted tensor B.
        \param kb Scalar transformation of B.
        \param kc Scalar transformation of result (C).
     **/
    ctf_tod_contract2_streamed(
        const contraction2<N, M, K> &contr,
        ctf_dense_tensor_i<NA, double> &ta,
        const scalar_transf<double> &ka,
        ctf_dense_tensor_i<NB, double> &tb,
        const scalar_transf<double> &kb,
        const scalar_transf<double> &kc = scalar_transf<double>());

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_contract2_streamed() { }

    /** \brief Adds a set of arguments to the argument list
        \param contr Contraction.
        \param ta First contracted tensor A.
        \param ka Scalar transformation of A.
        \param tb Second contracted tensor B.
        \param kb Scalar transformation of B.
        \param kc Scalar transformation of result (C).
     **/
    void add_args(
        const contraction2<N, M, K> &contr,
        ctf_dense_tensor_i<NA, double> &ta,
        const scalar_transf<double> &ka,
        ctf_dense_tensor_i<NB, double> &tb,
        const scalar_transf<double> &kb,
        const scalar_transf<double> &kc);

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tc Output tensor.
     **/
    void perform(
        bool zero,
        ctf_dense_tensor_i<NC, double> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_STREAMED_H
