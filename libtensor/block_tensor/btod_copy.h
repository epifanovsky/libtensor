#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_copy.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/bto/bto_stream_i.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief Copies a block tensor with an optional transformation
    \tparam N Tensor order.

    \sa gen_bto_copy

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_copy : public additive_bto<N, btod_traits>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_copy< N, btod_traits, btod_copy<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param c Scaling coefficient.
     **/
    btod_copy(
        block_tensor_rd_i<N, double> &bta,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(
            permutation<N>(), scalar_transf<double>(c))) {

    }

    /** \brief Initializes the operation
        \param bta Source block tensor (A).
        \param perma Permutation of A.
        \param c Scaling coefficient.
     **/
    btod_copy(
        block_tensor_i<N, double> &bta,
        const permutation<N> &perma,
        double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(perma, scalar_transf<double>(c))) {

    }

    virtual ~btod_copy() { }

    virtual const block_index_space<N> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(bto_stream_i<N, btod_traits> &out);

    virtual void perform(block_tensor_i<N, double> &btb);

    virtual void perform(
        block_tensor_i<N, double> &btb,
        const double &c);

    virtual void compute_block(
        dense_tensor_i<N, double> &blkb,
        const index<N> &ib);

    virtual void compute_block(
        bool zero,
        dense_tensor_i<N, double> &blkb,
        const index<N> &ib,
        const tensor_transf<N, double> &trb,
        const double &c);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COPY_H
