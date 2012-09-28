#ifndef LIBTENSOR_BTOD_APPLY_H
#define LIBTENSOR_BTOD_APPLY_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_apply.h>

namespace libtensor {


template<size_t N, typename Functor>
class btod_apply : public additive_bto<N, btod_traits>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;

private:
    gen_bto_apply<N, Functor, btod_traits, btod_apply<N, Functor> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Input block %tensor
        \param fn Functor
        \param c Scaling factor (applied before functor).
     **/
    btod_apply(block_tensor_i<N, double> &bta,
            const Functor &fn, double c = 1.0) :

        m_gbto(bta, fn, scalar_transf<double>(c)) {
    }

    /** \brief Initializes the operation
        \param bta Input block %tensor
        \param fn Functor
        \param p Permutation
        \param c Scaling factor (applied before functor).
     **/
    btod_apply(block_tensor_i<N, double> &bta, const Functor &fn,
            const permutation<N> &p, double c = 1.0) :
        m_gbto(bta, fn, scalar_transf<double>(c),
                tensor_transf<N, double>(p)) {
    }

    virtual ~btod_apply() { }

    virtual const block_index_space<N> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    //@}

    virtual void perform(gen_block_stream_i<N, bti_traits> &out) {

        m_gbto.perform(out);
    }

    virtual void perform(block_tensor_i<N, double> &btb);
    virtual void perform(block_tensor_i<N, double> &btb, const double &c);

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

#endif // LIBTENSOR_BTOD_APPLY_H

#include "impl/btod_apply_impl.h"

