#ifndef LIBTENSOR_BTO_COPY_H
#define LIBTENSOR_BTO_COPY_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/bto/additive_bto.h>

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
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_t;

    //! Type of blocks of a block tensor
    typedef typename Traits::template block_type<N>::type block_t;

    typedef tensor_transf<N, element_t> tensor_tr_t;

    typedef scalar_transf<element_t> scalar_tr_t;

public:
    static const char *k_clazz; //!< Class name

private:
    block_tensor_t &m_bta; //!< Source block %tensor
    tensor_tr_t m_tr; //!< Tensor transformation
    block_index_space<N> m_bis; //!< Block %index space of output
    dimensions<N> m_bidims; //!< Block %index dimensions
    symmetry<N, double> m_sym; //!< Symmetry of output
    assignment_schedule<N, double> m_sch;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the copy operation
        \param bt Source block %tensor.
        \param c Scaling coefficient.
     **/
    bto_copy(block_tensor_t &bta, const scalar_tr_t &c = scalar_tr_t());

    /** \brief Initializes the permuted copy operation
        \param bt Source block %tensor.
        \param p Permutation.
        \param c Scaling coefficient.
     **/
    bto_copy(block_tensor_t &bta, const permutation<N> &p,
            const scalar_tr_t &c = scalar_tr_t());

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

    virtual const symmetry<N, element_t> &get_symmetry() const {
        return m_sym;
    }

    using additive_bto<N, typename Traits::additive_bto_traits>::perform;

    virtual void sync_on();
    virtual void sync_off();

    //@}

    //!    \name Implementation of libtensor::btod_additive<N>
    //@{
    virtual const assignment_schedule<N, element_t> &get_schedule() const {
        return m_sch;
    }
    //@}

protected:
    virtual void compute_block(bool zero, block_t &blk, const index<N> &ib,
        const tensor_tr_t &tr, const element_t &c);

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
