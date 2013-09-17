#ifndef LIBTENSOR_BTOD_SYMCONTRACT3_H
#define LIBTENSOR_BTOD_SYMCONTRACT3_H

#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_symcontract3.h>

namespace libtensor {


/** \brief Contracts a train of three tensors
    \tparam N1 Order of first tensor less first contraction degree.
    \tparam N2 Order of second tensor less total contraction degree.
    \tparam N3 Order of third tensor less second contraction degree.
    \tparam K1 First contraction degree.
    \tparam K2 Second contraction degree.

    This operation follows the same algorithm as btod_contract3,
    except it applies an antisymmetrization filter to the intermediate.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
class btod_symcontract3 : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;

private:
    gen_bto_symcontract3<N1, N2, N3, K1, K2, btod_traits,
        btod_symcontract3<N1, N2, N3, K1, K2> > m_gbto;

public:
    /** \brief Initializes the contraction
        \param contr1 First contraction (A with B).
        \param contr2 Second contraction (AB with C).
        \param bta First tensor argument (A).
        \param btb Second tensor argument (B).
        \param permab Symmetrization permutation of AB.
        \param symmab Symmetrize (true) or antisymmetrize (false) AB.
        \param btc Third tensor argument (C).
     **/
    btod_symcontract3(
        const contraction2<N1, N2 + K2, K1> &contr1,
        const contraction2<N1 + N2, N3, K2> &contr2,
        block_tensor_i<N1 + K1, double> &bta,
        block_tensor_i<N2 + K1 + K2, double> &btb,
        const permutation<N1 + N2 + K2> &permab,
        bool symmab,
        block_tensor_i<N3 + K2, double> &btc);

    /** \brief Initializes the contraction
        \param contr1 First contraction (A with B).
        \param contr2 Second contraction (AB with C).
        \param bta First tensor argument (A).
        \param btb Second tensor argument (B).
        \param permab Symmetrization permutation of AB.
        \param symmab Symmetrize (true) or antisymmetrize (false) AB.
        \param btc Third tensor argument (C).
        \param kd Scaling coefficient.
     **/
    btod_symcontract3(
        const contraction2<N1, N2 + K2, K1> &contr1,
        const contraction2<N1 + N2, N3, K2> &contr2,
        block_tensor_i<N1 + K1, double> &bta,
        block_tensor_i<N2 + K1 + K2, double> &btb,
        const permutation<N1 + N2 + K2> &permab,
        bool symmab,
        block_tensor_i<N3 + K2, double> &btc,
        double kd);

    /** \brief Virtual destructor
     **/
    virtual ~btod_symcontract3() { }

    /** \brief Computes the contraction
     **/
    void perform(gen_block_stream_i<N1 + N2 + N3, bti_traits> &out);

    /** \brief Computes the contraction
     **/
    void perform(block_tensor_i<N1 + N2 + N3, double> &btd);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SYMCONTRACT3_H

