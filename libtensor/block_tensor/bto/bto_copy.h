#ifndef LIBTENSOR_BTO_COPY_H
#define LIBTENSOR_BTO_COPY_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include "bto_stream_i.h"

namespace libtensor {


/** \brief Makes a copy of a block %tensor, applying a permutation and
        a scaling coefficient
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits>
class bto_copy :
    public additive_bto<N, typename Traits::additive_bto_traits>,
    public timings< bto_copy<N, Traits> > {

public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;

    typedef bto_stream_i<N, Traits> bto_stream_type;
    typedef tensor_transf<N, element_type> tensor_transf_t;
    typedef scalar_transf<element_type> scalar_transf_t;

public:
    static const char *k_clazz; //!< Class name

private:
    block_tensor_type &m_bta; //!< Source block %tensor
    tensor_transf_t m_tr; //!< Tensor transformation
    block_index_space<N> m_bis; //!< Block %index space of output
    dimensions<N> m_bidims; //!< Block %index dimensions
    symmetry<N, double> m_sym; //!< Symmetry of output
    assignment_schedule<N, double> m_sch;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the copy operation
        \param bt Source block %tensor.
        \param tr Transformation.
     **/
    bto_copy(block_tensor_type &bta, const tensor_transf_t &tr =
            tensor_transf_t());

    /** \brief Initializes the copy operation
        \param bt Source block %tensor.
        \param c Element-wise transformation.
     **/
    bto_copy(block_tensor_type &bta, const scalar_transf_t &c);

    /** \brief Initializes the permuted copy operation
        \param bt Source block %tensor.
        \param p Permutation.
        \param c Element-wise transformation.
     **/
    bto_copy(block_tensor_type &bta, const permutation<N> &p,
            const scalar_transf_t &c = scalar_transf_t());

    /** \brief Virtual destructor
     **/
    virtual ~bto_copy() { }

    //@}

    //!    \name Implementation of
    //!        libtensor::direct_block_tensor_operation<N, double>
    //@{
    virtual const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<N, element_type> &get_symmetry() const {
        return m_sym;
    }

    virtual void perform(block_tensor_type &bt);
    virtual void perform(block_tensor_type &bt, const element_type &c);
    virtual void perform(bto_stream_type &out);

    using additive_bto<N, typename Traits::additive_bto_traits>::perform;

    virtual void sync_on();
    virtual void sync_off();

    //@}

    //!    \name Implementation of libtensor::btod_additive<N>
    //@{
    virtual const assignment_schedule<N, element_type> &get_schedule() const {
        return m_sch;
    }
    //@}

    virtual void compute_block(bool zero, block_type &blk, const index<N> &ib,
        const tensor_transf_t &tr, const element_type &c);

private:
    static block_index_space<N> mk_bis(const block_index_space<N> &bis,
        const permutation<N> &perm);

    void make_schedule();

private:
    bto_copy(const bto_copy<N, Traits>&);
    bto_copy<N, Traits> &operator=(const bto_copy<N, Traits>&);

};


} // namespace libtensor


#endif // LIBTENSOR_BTO_COPY_H
