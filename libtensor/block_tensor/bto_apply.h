#ifndef LIBTENSOR_BTO_APPLY_H
#define LIBTENSOR_BTO_APPLY_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_apply.h>

namespace libtensor {


template<size_t N, typename Functor, typename T>
class bto_apply :
    public additive_gen_bto<N, typename bto_traits<T>::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename bto_traits<T>::bti_traits bti_traits;

private:
    gen_bto_apply<N, Functor, bto_traits<T>, bto_apply<N, Functor, T> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bta Input block %tensor
        \param fn Functor
        \param c Scaling factor (applied before functor).
     **/
    bto_apply(block_tensor_i<N, T> &bta,
            const Functor &fn, T c = 1.0) :

        m_gbto(bta, fn, scalar_transf<T>(c)) {
    }

    /** \brief Initializes the operation
        \param bta Input block %tensor
        \param fn Functor
        \param p Permutation
        \param c Scaling factor (applied before functor).
     **/
    bto_apply(block_tensor_i<N, T> &bta, const Functor &fn,
            const permutation<N> &p, T c = 1.0) :
        m_gbto(bta, fn, scalar_transf<T>(c),
                tensor_transf<N, T>(p)) {
    }

    virtual ~bto_apply() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N, T> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, T> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<N, bti_traits> &out) {

        m_gbto.perform(out);
    }

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb);

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btb,
            const scalar_transf<T> &c);

    virtual void compute_block(
            bool zero,
            const index<N> &ib,
            const tensor_transf<N, T> &trb,
            dense_tensor_wr_i<N, T> &blkb);

    //@}

    /** \brief Function for compatability
     **/
    void perform(block_tensor_i<N, T> &btb, T c);
};

template<size_t N, typename Functor>
using btod_apply = bto_apply<N, Functor, double>;

} // namespace libtensor

#endif // LIBTENSOR_BTO_APPLY_H

#include "impl/bto_apply_impl.h"

