#ifndef LIBTENSOR_DIAG_BTOD_COPY_H
#define LIBTENSOR_DIAG_BTOD_COPY_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_copy.h>
#include <libtensor/diag_block_tensor/diag_block_tensor_i.h>
#include "diag_btod_traits.h"

namespace libtensor {


/** \brief Copies a block tensor with an optional transformation
    \tparam N Tensor order.

    \sa gen_bto_copy

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N>
class diag_btod_copy : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename diag_btod_traits::bti_traits bti_traits;

private:
    gen_bto_copy< N, diag_btod_traits, diag_btod_copy<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param c Scaling coefficient.
     **/
    diag_btod_copy(
        diag_block_tensor_rd_i<N, double> &bta,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(
            permutation<N>(), scalar_transf<double>(c))) {

    }

    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param perma Permutation of A.
        \param c Scaling coefficient.
     **/
    diag_btod_copy(
        diag_block_tensor_rd_i<N, double> &bta,
        const permutation<N> &perma,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(perma, scalar_transf<double>(c))) {

    }

    virtual ~diag_btod_copy() { }

    virtual const block_index_space<N> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(
        gen_block_stream_i<N, bti_traits> &out) {

        m_gbto.perform(out);
    }

    virtual void perform(
        diag_block_tensor_i<N, double> &btb);

    virtual void perform(
        diag_block_tensor_i<N, double> &btb,
        const double &c);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_COPY_H
