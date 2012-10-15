#ifndef LIBTENSOR_DIAG_BTOD_CONTRACT2_H
#define LIBTENSOR_DIAG_BTOD_CONTRACT2_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_contract2.h>
#include <libtensor/diag_block_tensor/diag_block_tensor_i.h>
#include "diag_btod_traits.h"

namespace libtensor {


/** \brief Computes the contraction of two diagonal tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    \sa gen_bto_contract2

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N, size_t M, size_t K>
class diag_btod_contract2 : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    typedef typename diag_btod_traits::bti_traits bti_traits;

private:
    gen_bto_contract2< N, M, K, diag_btod_traits, diag_btod_contract2<N, M, K> >
        m_gbto;

public:
    /** \brief Initializes the operation
        \param contr Contraction.
        \param bta First diagonal block tensor (A).
        \param btb First diagonal block tensor (B).
     **/
    diag_btod_contract2(
        const contraction2<N, M, K> &contr,
        diag_block_tensor_rd_i<NA, double> &bta,
        diag_block_tensor_rd_i<NB, double> &btb) :

        m_gbto(contr, bta, scalar_transf<double>(1.0), btb,
            scalar_transf<double>(1.0), scalar_transf<double>(1.0)) {

    }

    virtual ~diag_btod_contract2() { }

    virtual const block_index_space<NC> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<NC, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<NC, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(
        gen_block_stream_i<NC, bti_traits> &out) {

        m_gbto.perform(out);
    }

    virtual void perform(
        diag_block_tensor_i<NC, double> &btc);

    virtual void perform(
        diag_block_tensor_i<NC, double> &btc,
        const double &d);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_CONTRACT2_H
